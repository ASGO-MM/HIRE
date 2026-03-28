# import debugpy

# try:
#     debugpy.listen(("localhost", 5803))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

import argparse
import torch
import os
import json
import copy
from tqdm import tqdm

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

import math

import h5py
from torch.utils.data import DataLoader, DistributedSampler
from dci_data import DenseCaptionedDataset, dci_custom_collate_fn
from functools import partial


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

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def add_diffusion_noise(image_tensor, noise_step):
    num_steps = 1000  # Number of diffusion steps

    # decide beta in each step
    betas = torch.linspace(-6,6,num_steps)
    betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5

    # decide alphas in each step
    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, dim=0)
    alphas_prod_p = torch.cat([torch.tensor([1]).float(), alphas_prod[:-1]],0) # p for previous
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

    def q_x(x_0,t):
        noise = torch.randn_like(x_0)
        alphas_t = alphas_bar_sqrt[t]
        alphas_1_m_t = one_minus_alphas_bar_sqrt[t]
        return (alphas_t*x_0 + alphas_1_m_t*noise)

    noisy_image = image_tensor.clone()
    image_tensor_cd = q_x(noisy_image, noise_step) 

    return image_tensor_cd

def create_wrapped_tensor(pos_hs, neg_hs, pad_position, image_names, pos_file, neg_file, start_save=595):
    new_pos = {}
    new_neg = {}

    for b, image_name in enumerate(image_names):
        stack_pos=[]
        stack_neg=[]
        for layer_index, two_hs in pos_hs.items():
                stack_pos.append(pos_hs[layer_index][b][start_save:][pad_position[b]])
                stack_neg.append(neg_hs[layer_index][b][start_save:][pad_position[b]])
        stack_pos = torch.stack(stack_pos, dim=0).to(torch.float16)   # [layer_index, num_token, hidden_dim]
        stack_neg = torch.stack(stack_neg, dim=0).to(torch.float16)   # [layer_index, num_token, hidden_dim]

        new_pos[image_name] = stack_pos
        new_neg[image_name] = stack_neg

    for image_name, pos_dict in new_pos.items():
        image_name = image_name.split('/')[-1].split('.')[0]
        if image_name in pos_file.keys():
            continue
        pos_file.create_group(image_name)
        pos_file[image_name].create_dataset(image_name, data=pos_dict.cpu().numpy())
        pos_file.flush()

    for image_name, neg_dict in new_neg.items():
        image_name = image_name.split('/')[-1].split('.')[0]
        if image_name in neg_file.keys():
            continue
        neg_file.create_group(image_name)
        neg_file[image_name].create_dataset(image_name, data=neg_dict.cpu().numpy())
        neg_file.flush()


extracted_buffer = {}
def extract_attn_output_hook(module, args, layer_idx):
    """
    Hook function: intercept the input of self_attn.o_proj (i.e., attn_output)
    """
    extracted_buffer[layer_idx] = args[0].detach().cpu()

def register_hooks(model):
    hooks = []
    # Adapter for LLaVA/Llama cture
    layers = model.model.layers if hasattr(model, "model") else model.layers
    
    for i, layer in enumerate(layers):
        target_module = layer.self_attn.o_proj
        hook_fn = partial(extract_attn_output_hook, layer_idx=i)
        handle = target_module.register_forward_pre_hook(hook_fn)
        hooks.append(handle)
    return hooks


def eval_model(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name)
    hook_handles = register_hooks(model)

    dataset = DenseCaptionedDataset(args.data_path)
    sampler = CustomDistributedSampler(dataset, num_replicas=args.num_chunks, rank=args.chunk_idx, seed=args.seed, total_samples=args.data_size)
    dataloader = DataLoader(dataset, batch_size=10, sampler=sampler, collate_fn=dci_custom_collate_fn, drop_last=False)

    os.makedirs(args.hs_path, exist_ok=True)

    pos_file = h5py.File(os.path.join(args.hs_path, f'pos_hs_dci_{args.chunk_idx}.h5'), 'a')
    neg_file = h5py.File(os.path.join(args.hs_path, f'neg_hs_dci_{args.chunk_idx}.h5'), 'a')

    pad_token_id = tokenizer.pad_token_id
    PROMPT = "Please describe this image in detail."
    qs = DEFAULT_IMAGE_TOKEN + '\n' + PROMPT

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
    image_token_length = 576
    start_save=image_token_length+input_ids.shape[0]-1
    text_start=input_ids.shape[0]
    pos_caption=None

    for image, pos_caption, image_names, _ in tqdm(dataloader):  # dci
        qs = PROMPT
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        ids = [int(image_name.split('_')[-1].split('.')[0]) for image_name in image_names]
        if pos_caption is None:
            captions = [caption_dict[id] for id in ids]
        else:
            captions=pos_caption

        new_input_ids = []
        max_len = 0
        for caption in captions:
            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], caption)
            prompt = conv.get_prompt()
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').cuda()
            new_input_ids.append(input_ids)
            if input_ids.shape[0]>max_len:
                max_len = input_ids.shape[0]
   
        padded_input_ids = []
        for tmp_ids in new_input_ids:
            pad_len = max_len - tmp_ids.shape[0]
            if pad_len > 0:
                padding = torch.full(
                    (pad_len,), 
                    pad_token_id, 
                    dtype=tmp_ids.dtype, 
                    device=tmp_ids.device
                )
                tmp_ids = torch.cat([tmp_ids, padding], dim=0)
            padded_input_ids.append(tmp_ids)
        
        input_ids = torch.stack(padded_input_ids, dim=0)
        pad_position = (input_ids[:,text_start:]!=pad_token_id).cpu()
        attention_mask = (input_ids != pad_token_id).long().cuda()
        image_tensor = process_images(image, image_processor, model.config)
        
        with torch.no_grad():
            _ = model(input_ids,images=image_tensor.half().cuda(), attention_mask=attention_mask)
            pos_hs = copy.deepcopy(extracted_buffer)
        
        extracted_buffer.clear()
        image_tensor = add_diffusion_noise(image_tensor, 999)
        with torch.no_grad():
            _ = model(input_ids, images=image_tensor.half().cuda(), attention_mask=attention_mask)
            neg_hs = copy.deepcopy(extracted_buffer)
        
        create_wrapped_tensor(pos_hs, neg_hs, pad_position, image_names, pos_file, neg_file, start_save)

    pos_file.close()
    neg_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--hs-path", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-size", type=int, default=2000)
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    args = parser.parse_args()

    eval_model(args)
