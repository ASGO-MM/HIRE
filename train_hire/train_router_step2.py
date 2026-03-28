import os
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import h5py
from router import DPOAgent
import argparse
from torch.nn.utils.rnn import pad_sequence


class HSDataset(Dataset):
    def __init__(self, pos_data_path, neg_data_path):
        self.pos_data_paths = pos_data_path if isinstance(pos_data_path, list) else [pos_data_path]
        self.neg_data_paths = neg_data_path if isinstance(neg_data_path, list) else [neg_data_path]
        
        self.image_name_list = []
        self.pos_map = {}
        self.neg_map = {}

        for idx, path in enumerate(self.pos_data_paths):
            with h5py.File(path, 'r') as f:
                for key in f.keys():
                    self.image_name_list.append(key)
                    self.pos_map[key] = idx
                    
        for idx, path in enumerate(self.neg_data_paths):
            with h5py.File(path, 'r') as f:
                for key in f.keys():
                    if key not in self.pos_map: 
                        self.image_name_list.append(key)
                    self.neg_map[key] = idx

        self.image_name_list = list(set(self.image_name_list))
        
        self.f_pos_handles = None
        self.f_neg_handles = None
        
        print(f'Dataset init done. Total unique images: {len(self.image_name_list)}')

    def _init_hdf5_handles(self):
        self.f_pos_handles = [h5py.File(path, 'r') for path in self.pos_data_paths]
        self.f_neg_handles = [h5py.File(path, 'r') for path in self.neg_data_paths]

    def __len__(self):
        return len(self.image_name_list)
    
    def __getitem__(self, index):
        if self.f_pos_handles is None or self.f_neg_handles is None:
            self._init_hdf5_handles()

        image_name = self.image_name_list[index]
        pos_hs_dict = {}
        neg_hs_dict = {}
        
        if image_name in self.pos_map:
            f_idx = self.pos_map[image_name]
            f_pos = self.f_pos_handles[f_idx]
            pos_hs_group = f_pos[image_name]
            for key in pos_hs_group.keys():
                pos_hs_dict[key] = pos_hs_group[key][...]
                
        if image_name in self.neg_map:
            f_idx = self.neg_map[image_name]
            f_neg = self.f_neg_handles[f_idx]
            neg_hs_group = f_neg[image_name]
            for key in neg_hs_group.keys():
                neg_hs_dict[key] = neg_hs_group[key][...]
                
        return image_name, pos_hs_dict, neg_hs_dict
    
    def __del__(self):
        if hasattr(self, 'f_pos_handles') and self.f_pos_handles is not None:
            for f in self.f_pos_handles:
                f.close()
        if hasattr(self, 'f_neg_handles') and self.f_neg_handles is not None:
            for f in self.f_neg_handles:
                f.close()


def custom_collate_fn(batch):
    image_name, pos_hs_dict, neg_hs_dict = zip(*batch)
    return list(image_name), pos_hs_dict, neg_hs_dict


def train_ddp(rank, world_size, train_device, args):
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        rank=rank,
        world_size=world_size
    )
    torch.cuda.set_device(train_device[rank])
    device = torch.device(f'cuda:{train_device[rank]}')
    
    os.makedirs(args.save_path, exist_ok=True)
    agent = DPOAgent(hidden_dim=4096, latent_dim=args.h_dim, action_dim=2, p_dropout=args.p_dropout, device=device)
    dtype = agent.dtype

    ddp_actor = nn.parallel.DistributedDataParallel(agent.actor, device_ids=[train_device[rank]])
    agent.actor = ddp_actor
    optimizer = torch.optim.SGD(ddp_actor.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, momentum=0.9)
    # optimizer = torch.optim.AdamW(ddp_actor.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=args.eta_min)
    num_epochs=args.num_epochs
    accumulate_steps = args.accumulate_steps

    p_path, n_path = [], []
    p_list = os.listdir(args.data_path.replace('<type>', 'pos'))
    n_list = os.listdir(args.data_path.replace('<type>', 'neg'))
    for p_p, n_p in zip(p_list, n_list):
        p_path.append(os.path.join(args.data_path.replace('<type>', 'pos'), p_p))
        n_path.append(os.path.join(args.data_path.replace('<type>', 'neg'), n_p))

    dataset = HSDataset(p_path, n_path)
    print(len(dataset))
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=custom_collate_fn, sampler=sampler, num_workers=4)

    losses=[]
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)
        for idx, (image_names, pos_dict, neg_dict) in enumerate(dataloader):
            pos_states_list = [torch.from_numpy(d['states']).to(dtype) for d in pos_dict]
            pos_actions_list = [torch.from_numpy(d['actions']).to(torch.long) for d in pos_dict]
            neg_states_list = [torch.from_numpy(d['states']).to(dtype) for d in neg_dict]
            neg_actions_list = [torch.from_numpy(d['actions']).to(torch.long) for d in neg_dict]

            padded_pos_states = pad_sequence(pos_states_list, batch_first=True, padding_value=0.0).to(device)
            padded_pos_actions = pad_sequence(pos_actions_list, batch_first=True, padding_value=-1).to(device)
            
            padded_neg_states = pad_sequence(neg_states_list, batch_first=True, padding_value=0.0).to(device)
            padded_neg_actions = pad_sequence(neg_actions_list, batch_first=True, padding_value=-1).to(device)

            pos_logits = agent.actor(padded_pos_states)
            neg_logits = agent.actor(padded_neg_states)

            pos_log_probs = F.log_softmax(pos_logits, dim=-1)
            neg_log_probs = F.log_softmax(neg_logits, dim=-1)

            p_lp = pos_log_probs.gather(2, padded_pos_actions.unsqueeze(-1).clamp(min=0)).squeeze(-1)
            n_lp = neg_log_probs.gather(2, padded_neg_actions.unsqueeze(-1).clamp(min=0)).squeeze(-1)

            pos_mask = (padded_pos_actions != -1).float()
            neg_mask = (padded_neg_actions != -1).float()

            log_p_preferred = (p_lp * pos_mask).sum(dim=1)
            log_p_non_preferred = (n_lp * neg_mask).sum(dim=1)

            dpo_loss = -F.logsigmoid(args.beta * (log_p_preferred - log_p_non_preferred)).mean()

            # ---------------------------------------------------------
            pos_probs = F.softmax(pos_logits, dim=-1)  # [B, S, 2]
            neg_probs = F.softmax(neg_logits, dim=-1)  # [B, S, 2]
            
            all_probs = torch.cat([pos_probs, neg_probs], dim=0)
            all_masks = torch.cat([pos_mask, neg_mask], dim=0)
            
            valid_token_count = all_masks.sum()
            
            if valid_token_count > 0:
                avg_action_probs = (all_probs * all_masks.unsqueeze(-1)).sum(dim=(0, 1)) / valid_token_count
                target_dist = torch.tensor([0.5, 0.5], device=device, dtype=avg_action_probs.dtype)

                log_avg_probs = torch.log(avg_action_probs + 1e-8)
                balance_loss = F.kl_div(log_avg_probs, target_dist, reduction='sum')
            else:
                balance_loss = torch.tensor(0.0, device=device)
            loss = dpo_loss + 0.05 * balance_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if rank == 0:
                losses.append(loss.item())
                print(f"Epoch {epoch} Step {idx} | Loss: {loss.item():.6f}")

        scheduler.step()
        if rank==0 and ((epoch+1) % num_epochs==0 or (epoch+1)%args.save_epoch==0):
            torch.save({
                'state_dict': agent.actor.module.state_dict() if hasattr(agent.actor, 'module') else agent.actor.state_dict(),
                'epoch': epoch,
                'loss': losses,
            }, os.path.join(args.save_path, f"router_llava_epoch{epoch+1}_new_sgd_banlance.pth"))
    dist.barrier()
    dist.destroy_process_group()

def get_args():
    parser = argparse.ArgumentParser(description="DPO Router Training for HIRE")

    # ----- Paths -----
    group_path = parser.add_argument_group('Paths')
    group_path.add_argument("--data-path", type=str, default="hs_data/agent_<type>_llava_1")
    group_path.add_argument("--save-path", type=str, default="output/RL_router")

    # ----- Hyperparameters -----
    group_train = parser.add_argument_group('Training')
    group_train.add_argument("--num-epochs", type=int, default=20)
    group_train.add_argument("--save-epoch", type=int, default=5)
    group_train.add_argument("--batch-size", type=int, default=1)
    group_train.add_argument("--learning-rate", type=float, default=1e-2)
    group_train.add_argument("--eta-min", type=float, default=1e-3)
    group_train.add_argument("--beta", type=float, default=0.1)
    group_train.add_argument("--weight-decay", type=float, default=1e-2)
    group_train.add_argument("--seed", type=int, default=42)
    group_train.add_argument("--accumulate-steps", type=int, default=1)

    # ----- Architecture -----
    group_model = parser.add_argument_group('Model')
    group_model.add_argument("--h_dim", type=int, nargs='+', default=[2048, 1024])
    group_model.add_argument("--hidden_dim", type=int, default=4096)
    group_train.add_argument("--p-dropout", type=float, default=0.5)

    # ----- Distributed -----
    group_dist = parser.add_argument_group('Distributed')
    group_dist.add_argument("--master_port", type=str, default="12125")
    group_dist.add_argument("--gpus", type=int, nargs='+', default=[0, 1, 2, 3, 4, 5, 6, 7])

    return parser.parse_args()

def main():
    args = get_args()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.master_port

    train_device = args.gpus
    world_size = len(train_device)
    print(f"world_size :{world_size}")

    if world_size > 1:
        mp.spawn(train_ddp, args=(world_size, train_device, args), nprocs=world_size, join=True)
    else:
        train_ddp(0, 1, train_device, args)

if __name__ == "__main__":
    main()