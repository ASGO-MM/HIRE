import os
import numpy as np
import argparse
from truthx import TruthX
from tqdm import tqdm
import h5py
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.distributed as dist
import random
import torch.nn.functional as F
import torch.multiprocessing as mp


def pack_final_model(args):
    print("\n" + "="*50)
    print("Starting Final Model Packing...")
    print("="*50)

    pos_path = os.path.join(args.direction_save_path, 'ni_pos_center.npy')
    neg_path = os.path.join(args.direction_save_path, 'ni_neg_center.npy')
    
    last_epoch = args.epoch - 1
    checkpoint_path = os.path.join(args.checkpoint_path, f"model_epoch_{last_epoch}_finished.pth")

    print(f"Loading features from {args.direction_save_path}...")
    tru_pos_hs = np.load(pos_path)
    tru_neg_hs = np.load(neg_path)
    
    pos_center = np.mean(tru_pos_hs, axis=1)
    neg_center = np.mean(tru_neg_hs, axis=1)

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    train_layer = checkpoint.get('train_layer', None)
    if train_layer is None:
        train_layer = list(range(0, 63, 2))
    
    pos_center_dict = {}
    neg_center_dict = {}

    for idx, layer_index in enumerate(train_layer):
        pos_center_dict[layer_index] = torch.from_numpy(pos_center[idx])
        neg_center_dict[layer_index] = torch.from_numpy(neg_center[idx])

    for layer_index in pos_center_dict.keys():
        assert not (pos_center_dict[layer_index] == neg_center_dict[layer_index]).all(), \
            f"Layer {layer_index}: Positive and Negative centers are identical!"

    checkpoint['pos_center'] = pos_center_dict
    checkpoint['neg_center'] = neg_center_dict

    final_save_name = os.path.join(args.checkpoint_path, 'hire_editor_final.pth')
    torch.save(checkpoint, final_save_name)
    
    print(f"Success! Final model packed to: {final_save_name}")
    print("="*50 + "\n")

class HSDataset(Dataset):
    def __init__(self, pos_data_path, neg_data_path):
        if isinstance(pos_data_path, list):
            self.f_pos = [h5py.File(path, 'r') for path in pos_data_path]
        else:
            self.f_pos = [h5py.File(pos_data_path, 'r')]
        
        if isinstance(neg_data_path, list):
            self.f_neg = [h5py.File(path, 'r') for path in neg_data_path]
        else:
            self.f_neg = [h5py.File(neg_data_path, 'r')]
        
        self.image_name_list = []
        for f in self.f_pos:
            self.image_name_list += list(f.keys())
    
    def __len__(self):
        return len(self.image_name_list)
    
    def __getitem__(self, index):
        image_name = self.image_name_list[index]
        
        pos_hs_dict = {}
        for f in self.f_pos:
            if image_name in f:
                pos_hs_group = f[image_name]
                for key in pos_hs_group:
                    pos_hs_dict[key] = pos_hs_group[key][...]
        
        neg_hs_dict = {}
        for f in self.f_neg:
            if image_name in f:
                neg_hs_group = f[image_name]
                for key in neg_hs_group:
                    neg_hs_dict[key] = neg_hs_group[key][...]
        
        return image_name, pos_hs_dict, neg_hs_dict
    
    def _merge_data(self, existing_data, new_data):
        return np.concatenate((existing_data, new_data), axis=0)

    def __del__(self):
        if hasattr(self, 'f_pos'):
            if isinstance(self.f_pos, list):
                for f in self.f_pos:
                    f.close()
            else:
                self.f_pos.close()
        if hasattr(self, 'f_neg'):
            if isinstance(self.f_neg, list):
                for f in self.f_neg:
                    f.close()
            else:
                self.f_neg.close()


def custom_collate_fn(batch):
    image_name, pos_hs_dict, neg_hs_dict = zip(*batch)
    return list(image_name), pos_hs_dict, neg_hs_dict


class CustomDistributedSampler(DistributedSampler):
    def __init__(self, dataset, num_replicas, rank, shuffle=False, seed=0, total_samples=500):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed)
        self.total_samples = total_samples

    def __iter__(self):
        indices = list(super().__iter__())

        num_samples_per_replica = self.total_samples // self.num_replicas
        remainder = self.total_samples % self.num_replicas
        if self.rank < remainder:
            num_samples_per_replica += 1

        indices = indices[:num_samples_per_replica]
        return iter(indices)

    def __len__(self):

        num_samples_per_replica = self.total_samples // self.num_replicas
        remainder = self.total_samples % self.num_replicas
        if self.rank < remainder:
            num_samples_per_replica += 1
        return num_samples_per_replica


def compute_semantic_loss(pos_sample, neg_sample, ae_model):
    temperature = 0.1
    encoded_pos = ae_model.get_semantic_latent_rep(pos_sample) # [L, S, D]
    encoded_neg = ae_model.get_semantic_latent_rep(neg_sample) # [L, S, D]
    
    pos_norm = F.normalize(encoded_pos, p=2, dim=-1)
    neg_norm = F.normalize(encoded_neg, p=2, dim=-1)

    pos_fenzi = torch.exp((pos_norm * neg_norm).sum(dim=-1) / temperature)
    
    pos_pos_sim = torch.matmul(pos_norm, pos_norm.transpose(-1, -2)) / temperature
    mask = torch.eye(pos_norm.shape[1], device=pos_sample.device).bool().unsqueeze(0)
    pos_pos_sim = pos_pos_sim.masked_fill(mask, -float('inf'))
    pos_fenmu = torch.exp(pos_pos_sim).sum(dim=-1) # [L, S]
    loss = -torch.log(pos_fenzi / (pos_fenzi + pos_fenmu)).mean()

    neg_fenzi = torch.exp((neg_norm * pos_norm).sum(dim=-1) / temperature)
    neg_neg_sim = torch.matmul(neg_norm, neg_norm.transpose(-1, -2)) / temperature
    neg_neg_sim = neg_neg_sim.masked_fill(mask, -float('inf'))
    neg_fenmu = torch.exp(neg_neg_sim).sum(dim=-1)
    
    loss += -torch.log(neg_fenzi / (neg_fenzi + neg_fenmu)).mean()

    return loss / 2

def compute_truthful_loss(pos_sample, neg_sample, ae_model):
    temperature = 0.1
    encoded_pos = ae_model.get_truthful_latent_rep(pos_sample) # [L, S, D]
    encoded_neg = ae_model.get_truthful_latent_rep(neg_sample) # [L, S, D]
    
    pos_norm = F.normalize(encoded_pos, p=2, dim=-1)
    neg_norm = F.normalize(encoded_neg, p=2, dim=-1)
    mask = torch.eye(pos_norm.shape[1], device=pos_sample.device).bool().unsqueeze(0)

    pos_pos_sim = torch.matmul(pos_norm, pos_norm.transpose(-1, -2)) / temperature
    pos_pos_sim = pos_pos_sim.masked_fill(mask, -float('inf'))
    pos_fenzi = torch.exp(pos_pos_sim).sum(dim=-1) # [L, S]
    
    pos_neg_sim = torch.matmul(pos_norm, neg_norm.transpose(-1, -2)) / temperature
    pos_fenmu = torch.exp(pos_neg_sim).sum(dim=-1) # [L, S]
    
    pos_loss = -torch.log(pos_fenzi / (pos_fenzi + pos_fenmu)).mean()

    neg_neg_sim = torch.matmul(neg_norm, neg_norm.transpose(-1, -2)) / temperature
    neg_neg_sim = neg_neg_sim.masked_fill(mask, -float('inf'))
    neg_fenzi = torch.exp(neg_neg_sim).sum(dim=-1)
    
    neg_pos_sim = torch.matmul(neg_norm, pos_norm.transpose(-1, -2)) / temperature
    neg_fenmu = torch.exp(neg_pos_sim).sum(dim=-1)
    
    neg_loss = -torch.log(neg_fenzi / (neg_fenzi + neg_fenmu)).mean()

    return (pos_loss + neg_loss) / 2


def compute_recon_loss(pos_sample, neg_sample, ae_model):
    l, s, d = pos_sample.shape
    pos_recon = ae_model(pos_sample.reshape(-1, d))[0].reshape(l,s,d)
    neg_recon = ae_model(neg_sample.reshape(-1, d))[0].reshape(l,s,d)
    loss = F.mse_loss(pos_recon, pos_sample) + F.mse_loss(neg_recon, neg_sample)
    return loss / 2


def compute_edit_loss(pos_samples, neg_samples, ae_model):
    edit_loss = 0
    num_layer, s_len, dim = pos_samples.shape
    
    tru_neg_hs = ae_model.get_truthful_latent_rep(neg_samples)
    edit_neg = ae_model(pos_samples.reshape(-1, dim), tru_neg_hs)[0]
    loss = F.mse_loss(edit_neg.reshape(num_layer, s_len, dim), neg_samples.detach(), reduction='mean')
    edit_loss += loss

    tru_pos_hs = ae_model.get_truthful_latent_rep(pos_samples)
    edit_pos = ae_model(neg_samples.reshape(-1, dim), tru_pos_hs)[0]
    loss = F.mse_loss(edit_pos.reshape(num_layer, s_len, dim), pos_samples.detach(), reduction='mean')
    edit_loss += loss

    return edit_loss / 2


def train_ddp(rank, args):
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    dist.init_process_group('nccl', rank=rank, world_size=args.num_chunks)
    
    os.makedirs(args.checkpoint_path, exist_ok=True)    
    pos_file = [os.path.join(args.hidden_states_path, f"pos_hs_{args.type}_{i}.h5") for i in range(4)]
    neg_file = [os.path.join(args.hidden_states_path, f"neg_hs_{args.type}_{i}.h5") for i in range(4)]

    batch_size = 1
    learning_rate = args.learning_rate
    iter_show = 10
    total_epoch = args.epoch
    
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    editor = TruthX(None, args.h_dim, args.i_dim, train_layer=args.edit_layer, device=device)
    editor.is_train = True
    editor.ae_model.train()
    extract_layer_index = [int(layer / 2) for layer in args.edit_layer]

    dataset = HSDataset(pos_file, neg_file)
    sampler = CustomDistributedSampler(dataset, num_replicas=args.num_chunks, rank=rank, shuffle=True, seed=args.seed, total_samples=args.data_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=custom_collate_fn)

    accumulate_steps = args.accumulate_steps

    ddp_ae_model = torch.nn.parallel.DistributedDataParallel(
        editor.ae_model,
        device_ids=[rank],
        find_unused_parameters=False,
    )
    editor.ae_model = ddp_ae_model
    ae_model = ddp_ae_model.module

    optimizer = torch.optim.SGD(ddp_ae_model.parameters(), lr=learning_rate, weight_decay=args.weight_decay, momentum=0.99)
    if args.min_lr:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epoch, eta_min=args.min_lr)
    
    record_loss = {'total_loss':[], 'recon_loss':[], 'sem_loss':[], 'edi_loss':[], 'tru_loss':[]}

    for epoch in range(total_epoch):
        sampler.set_epoch(epoch)
        iter_loss = 0.0
        r_loss=0.0
        s_loss=0.0
        e_loss=0.0
        t_loss=0.0
        for iter_idx, (image_names, pos_hs_dicts, neg_hs_dicts) in enumerate(dataloader):
            pos_hs_dicts = pos_hs_dicts[0]
            neg_hs_dicts = neg_hs_dicts[0]
            
            pos_samples = [torch.from_numpy(pos_hs_dicts[image_id][extract_layer_index]).type_as(ae_model.semantic_encoder[0][0].weight).to(device) for image_id in pos_hs_dicts.keys()]  # [layer_num, sequence_length, hidden_dim]
            neg_samples = [torch.from_numpy(neg_hs_dicts[image_id][extract_layer_index]).type_as(ae_model.semantic_encoder[0][0].weight).to(device) for image_id in neg_hs_dicts.keys()]  # [layer_num, sequence_length, hidden_dim]

            loss = 0
            for pos_sample, neg_sample in zip(pos_samples, neg_samples):
                max_len = args.max_len
                if pos_sample.shape[1] > max_len:
                    indices = torch.randperm(pos_sample.shape[1])[:max_len]    
                    indices = indices.sort()[0]
                    pos_sample = pos_sample[:, indices, :]
                    neg_sample = neg_sample[:, indices, :]
                
                sem_loss = compute_semantic_loss(pos_sample, neg_sample, ae_model)
                tru_loss = compute_truthful_loss(pos_sample, neg_sample, ae_model)
                edi_loss = compute_edit_loss(pos_sample, neg_sample, ae_model)
                rec_loss = compute_recon_loss(pos_sample, neg_sample, ddp_ae_model)
                loss += sem_loss + tru_loss + 0.5 * (edi_loss + rec_loss)
            loss = loss / batch_size 

            loss.backward()
            if (iter_idx+1) % accumulate_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                
            if dist.get_rank() == 0:
                print(f"rank:{rank}, iter:{iter_idx}, loss:{loss.item()}, recon_loss:{rec_loss.item()}, seman_loss:{sem_loss.item()}")

            iter_loss += loss.item()
            r_loss += rec_loss.item()
            s_loss += sem_loss.item()
            t_loss += tru_loss.item()
            e_loss += edi_loss.item()
            if (iter_idx + 1) % iter_show == 0:
                avg_loss = iter_loss / iter_show
                record_loss['total_loss'].append(avg_loss)
                record_loss['recon_loss'].append(r_loss/iter_show)
                record_loss['sem_loss'].append(s_loss/iter_show)
                record_loss['tru_loss'].append(t_loss/iter_show)
                record_loss['edi_loss'].append(e_loss/iter_show)
                if dist.get_rank() == 0:
                    print(f"Epoch [{epoch}/{total_epoch}], Iter [{iter_idx}/{len(dataloader)}], Loss: {avg_loss}")
                iter_loss = 0.0
                r_loss = 0.0
                s_loss = 0.0
                e_loss = 0.0
                t_loss = 0.0
        
        if (iter_idx + 1) % accumulate_steps != 0:
            optimizer.step()
            optimizer.zero_grad()
        scheduler.step()

        if dist.get_rank() == 0:    
            checkpoint_name = os.path.join(args.checkpoint_path, f"model_epoch_{epoch}_finished.pth")
            torch.save({
                'epoch': epoch,
                'record_loss': record_loss,
                'image_iter': -1,
                'state_dict': ae_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'train_layer': editor.train_layer,
            }, checkpoint_name)
    dist.barrier()

    # -----------------------
    # Extraction Direction
    # -----------------------
    if rank == 0:
        print("\n" + "="*50)
        print("Training finished. Starting Direction Extraction...")
        print("="*50)
    
    editor.is_train = False
    ae_model.eval() 
    os.makedirs(args.direction_save_path, exist_ok=True)

    extract_sampler = DistributedSampler(dataset, num_replicas=args.num_chunks, rank=rank, shuffle=True, seed=args.seed)
    extract_dataloader = DataLoader(dataset, batch_size=1, sampler=extract_sampler, collate_fn=custom_collate_fn, drop_last=True)

    if rank == 0:
        print(f"Extracting features to {args.direction_save_path} ...")

    all_pos_feats = []
    all_neg_feats = []
    
    for iter_idx, (image_names, pos_hs_dicts, neg_hs_dicts) in enumerate(tqdm(extract_dataloader, desc=f"Rank {rank} Extracting", dynamic_ncols=True)):
        pos_hs_dicts = pos_hs_dicts[0]
        neg_hs_dicts = neg_hs_dicts[0]

        pos_sample = [torch.from_numpy(pos_hs_dicts[image_id][extract_layer_index]).type_as(ae_model.semantic_encoder[0][0].weight).to(device) for image_id in pos_hs_dicts.keys()][0]
        neg_sample = [torch.from_numpy(neg_hs_dicts[image_id][extract_layer_index]).type_as(ae_model.semantic_encoder[0][0].weight).to(device) for image_id in neg_hs_dicts.keys()][0]

        with torch.no_grad():
            tru_pos_hs = ddp_ae_model.module.encode_truthful(pos_sample).cpu()
            tru_neg_hs = ddp_ae_model.module.encode_truthful(neg_sample).cpu()
        
        all_pos_feats.append(tru_pos_hs.mean(dim=1, keepdim=True))
        all_neg_feats.append(tru_neg_hs.mean(dim=1, keepdim=True))

    if all_pos_feats:
        all_pos_feats = torch.cat(all_pos_feats, dim=1).numpy()
        all_neg_feats = torch.cat(all_neg_feats, dim=1).numpy()
        
        pos_save_file = os.path.join(args.direction_save_path, f'pos_{rank}.npy')
        neg_save_file = os.path.join(args.direction_save_path, f'neg_{rank}.npy')
        
        np.save(pos_save_file, all_pos_feats)
        np.save(neg_save_file, all_neg_feats)
        print(f"Rank {rank}: Saved temporary files.")
    else:
        print(f"Rank {rank}: No data extracted.")

    if rank == 0:
        print("All ranks finished extraction. Combining files...")
        combined_pos = []
        combined_neg = []
        
        for r in range(args.num_chunks):
            p_path = os.path.join(args.direction_save_path, f'pos_{r}.npy')
            n_path = os.path.join(args.direction_save_path, f'neg_{r}.npy')
            
            if os.path.exists(p_path) and os.path.exists(n_path):
                combined_pos.append(np.load(p_path))
                combined_neg.append(np.load(n_path))
            else:
                print(f"Warning: Missing files for rank {r}")

        if combined_pos and combined_neg:
            final_pos = np.concatenate(combined_pos, axis=1)
            final_neg = np.concatenate(combined_neg, axis=1)
            
            np.save(os.path.join(args.direction_save_path, 'ni_pos_center.npy'), final_pos)
            np.save(os.path.join(args.direction_save_path, 'ni_neg_center.npy'), final_neg)
            print(f"Success! Final files saved to {args.direction_save_path}")
            
            for r in range(args.num_chunks):
                p_path = os.path.join(args.direction_save_path, f'pos_{r}.npy')
                n_path = os.path.join(args.direction_save_path, f'neg_{r}.npy')
                if os.path.exists(p_path): os.remove(p_path)
                if os.path.exists(n_path): os.remove(n_path)
            print("Temporary files cleaned up.")
        else:
            print("Error: No data to combine.")

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-states-path", type=str, default="")
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--min-lr", type=float, default=1e-3)
    parser.add_argument("--accumulate-steps", type=int, default=4)
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-size", type=int, default=1000)
    parser.add_argument("--max-len", type=int, default=100)
    parser.add_argument("--checkpoint-path", type=str, default="")
    parser.add_argument("--direction-save-path", type=str, default="")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--type", type=str, required=True)

    parser.add_argument('--h-dim', type=int, default=4096)
    parser.add_argument('--i-dim', type=int, nargs='+', default=[2048, 1024])
    parser.add_argument('--edit-layer', type=int, nargs='+', default=list(range(0, 63, 2)))
    args = parser.parse_args()

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12194'  # Use a free port

    if args.num_chunks!=1:
        mp.spawn(train_ddp, args=(args,), nprocs=args.num_chunks, join=True)
    else:
        train_ddp(0, args)

    pack_final_model(args)