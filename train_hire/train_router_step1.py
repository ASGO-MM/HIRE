import argparse
import torch
import os
import h5py
import pickle
import numpy as np
from tqdm import tqdm
from functools import partial

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from torch.utils.data import DataLoader, DistributedSampler
from coco_data import CocoDataset, custom_collate_fn
from truthx import TruthX
from CHAIR import CHAIR

# ===========================
# Global Buffer for recording generated trajectories
# ===========================
# Structure: { 'states': [step_0_states, step_1_states, ...], 'actions': [...] }
# Each step_n_states has dimensions [group_size, dim]
TRAJECTORY_BUFFER = {"states": [], "actions": []}
# Records the current step's action sampled at Layer 1 for subsequent layers
CURRENT_STEP_ACTION = None

def clear_buffer():
    global TRAJECTORY_BUFFER, CURRENT_STEP_ACTION
    TRAJECTORY_BUFFER = {"states": [], "actions": []}
    CURRENT_STEP_ACTION = None

def hire_router_hook(module, args, layer_idx, editor, group_size):
    global CURRENT_STEP_ACTION, TRAJECTORY_BUFFER
    hidden_states = args[0] # [batch_size, seq_len, dim]
        
    if layer_idx == 0:
        current_states = hidden_states[:, -1, :].detach().cpu().float().numpy()
        
        current_actions = torch.randint(0, 2, (hidden_states.shape[0],), device=hidden_states.device)
        
        TRAJECTORY_BUFFER["states"].append(current_states)
        TRAJECTORY_BUFFER["actions"].append(current_actions.cpu().numpy())
        
        CURRENT_STEP_ACTION = current_actions

    if layer_idx*2 in editor.train_layer and CURRENT_STEP_ACTION is not None:
        mask = CURRENT_STEP_ACTION.bool()
        
        editor.cur_layer_id = f"{layer_idx}.attn"
        edited_states = editor.edit(hidden_states)
        
        hidden_states[mask] = edited_states[mask]

    return (hidden_states,)

def register_router_hooks(model, editor, group_size):
    hooks = []
    layers = model.model.layers if hasattr(model, "model") else model.layers
    for i, layer in enumerate(layers):
        target_module = layer.self_attn.o_proj
        hook_fn = partial(hire_router_hook, layer_idx=i, editor=editor, group_size=group_size)
        handle = target_module.register_forward_pre_hook(hook_fn)
        hooks.append(handle)
    return hooks

def save_trajectories(image_name, pos_file, neg_file, best_idx, worst_idx, chair_results):
    states_all = np.stack(TRAJECTORY_BUFFER["states"], axis=0) 
    actions_all = np.stack(TRAJECTORY_BUFFER["actions"], axis=0)

    pos_states = states_all[:, best_idx, :].astype(np.float16)
    pos_actions = actions_all[:, best_idx].astype(np.int32)
    
    neg_states = states_all[:, worst_idx, :].astype(np.float16)
    neg_actions = actions_all[:, worst_idx].astype(np.int32)

    for f, s, a, m in zip([pos_file, neg_file], [pos_states, neg_states], [pos_actions, neg_actions], [chair_results['best'], chair_results['worst']]):
        if image_name not in f:
            grp = f.create_group(image_name)
            grp.create_dataset('states', data=s)
            grp.create_dataset('actions', data=a)
            for k, v in m.items():
                grp.create_dataset(k, data=np.array([v]))


def eval_model(args):
    disable_torch_init()
    device = f"cuda:{args.chunk_idx}"

    editor = TruthX(args.editor_model_path, args.h_dim, args.i_dim, train_layer=args.edit_layer, device=device)
    editor.eval()
    
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name, device_map=device, device=device)
    register_router_hooks(model, editor, args.group_size)

    dataset = CocoDataset(args.image_folder, args.anno_folder, subset_size=args.data_size, seed=args.seed)
    sampler = DistributedSampler(dataset, num_replicas=args.num_chunks, rank=args.chunk_idx, seed=args.seed, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, collate_fn=custom_collate_fn)

    pos_path = args.pos_path
    neg_path = args.neg_path
    os.makedirs(pos_path, exist_ok=True); os.makedirs(neg_path, exist_ok=True)
    
    pos_file = h5py.File(f'{pos_path}/temp_pos_hs_{args.chunk_idx}.h5', 'a')
    neg_file = h5py.File(f'{neg_path}/temp_neg_hs_{args.chunk_idx}.h5', 'a')
    
    chair_evaluator = pickle.load(open(args.chair_path, 'rb'))

    for image, image_names, _ in tqdm(dataloader):
        qs = DEFAULT_IMAGE_TOKEN + "\nPlease describe this image in detail."
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs); conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)
        input_ids = input_ids.repeat(args.group_size, 1)
        image_tensor = process_images(image, image_processor, model.config).repeat(args.group_size, 1, 1, 1).to(device, dtype=torch.float16)

        clear_buffer()
        with torch.inference_mode():
            outputs_gen = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=False,
                max_new_tokens=256,
                use_cache=True,
                return_dict_in_generate=True
            )

        gen_ids = outputs_gen.sequences[:, input_ids.shape[1]:]
        gen_texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        img_id_int = int(image_names[0].split('_')[-1].split('.')[0])
        
        chair_info = chair_evaluator.compute_score(gen_texts, [img_id_int] * args.group_size, "caption")
        scores = [info['metrics']['CHAIRi'] for info in chair_info['sentences']]
        
        best_idx, worst_idx = scores.index(min(scores)), scores.index(max(scores))

        if scores[best_idx] != scores[worst_idx]:

            m = chair_info['sentences']
            chair_results = {
                'best': {'chair_s': m[best_idx]['metrics']['CHAIRs'], 'chair_i': m[best_idx]['metrics']['CHAIRi'], 'recall': m[best_idx]['metrics']['Recall']},
                'worst': {'chair_s': m[worst_idx]['metrics']['CHAIRs'], 'chair_i': m[worst_idx]['metrics']['CHAIRi'], 'recall': m[worst_idx]['metrics']['Recall']}
            }

            save_trajectories(image_names[0], pos_file, neg_file, best_idx, worst_idx, chair_results)

    pos_file.close(); neg_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Paths
    parser.add_argument("--editor-model-path", type=str, default="")
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--anno-folder", type=str, default="")
    parser.add_argument("--output-file", type=str, default="output/gen_cap.json")
    parser.add_argument("--chair-path", type=str, default="Chair2017.pkl", help="Path to Chair2017.pkl")
    parser.add_argument("--hs-path", type=str, default="", help="Directory to save H5 files")
    parser.add_argument("--pos-path", type=str, required=True, help="Directory to save H5 files")
    parser.add_argument("--neg-path", type=str, required=True, help="Directory to save H5 files")
    
    # Settings
    parser.add_argument("--h-dim", type=int, default=4096)
    parser.add_argument("--i-dim", type=int, nargs='+', default=[2048, 1024])
    parser.add_argument("--edit-layer", type=int, nargs='+', default=list(range(0,63,2)))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-size", type=int, default=2000)
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1, help="Keep 1 for this script")
    parser.add_argument("--group-size", type=int, default=10, help="Number of samples per image")
    
    args = parser.parse_args()

    eval_model(args)