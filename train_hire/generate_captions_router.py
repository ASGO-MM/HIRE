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
from router import DPOAgent
from filelock import FileLock

CURRENT_STEP_ACTION = None

def editor_router_hook(module, args, layer_idx, editor, agent=None):
    global CURRENT_STEP_ACTION
    hidden_states = args[0]

    if layer_idx == 0:
        ori_dtype = hidden_states.dtype
        state_input = hidden_states[:,-1,:].detach().float()
        
        if agent is None:
            actions = torch.ones((hidden_states.shape[0],), dtype=ori_dtype, device=hidden_states.device)
        else:
            with torch.no_grad():
                logits = agent.actor(state_input)
                actions = torch.argmax(logits, dim=-1).to(ori_dtype) # [batch]
                print(actions)
            
        CURRENT_STEP_ACTION = actions

    if editor is not None and getattr(editor, "training", True) is False:
        if layer_idx*2 in editor.train_layer and CURRENT_STEP_ACTION is not None:
            edit_mask = CURRENT_STEP_ACTION.bool()
            
            editor.cur_layer_id = f"{layer_idx}.attn"
            edited_states = editor.edit(hidden_states)
            
            hidden_states[edit_mask] = edited_states[edit_mask]

    return (hidden_states,)

def register_router_hooks(model, editor, agent):
    hooks = []
    layers = model.model.layers if hasattr(model, "model") else model.layers
    for i, layer in enumerate(layers):
        target_module = layer.self_attn.o_proj
        hook_fn = partial(editor_router_hook, layer_idx=i, editor=editor, agent=agent)
        handle = target_module.register_forward_pre_hook(hook_fn)
        hooks.append(handle)
    print(f"Success: Registered Router-controlled editor hooks to {len(hooks)} layers.")
    return hooks

def eval_model(args):
    disable_torch_init()
    device = "cuda:0"

    train_layer = list(range(0, 63, 2))
    editor = TruthX(args.editor_path, 4096, [2048, 1024], train_layer=train_layer, device=device)
    editor.eval()

    if args.router_path:
        agent = DPOAgent(hidden_dim=4096, latent_dim=[2048, 1024], action_dim=2, device=device)
        checkpoint = torch.load(args.router_path, map_location=device)
        agent.actor.load_state_dict(checkpoint['state_dict'])
        print(f"Loaded Router from {args.router_path}")
        agent.eval()
    else:
        agent=None

    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name, device=device, device_map=device)
    register_router_hooks(model, editor, agent)

    dataset = CocoDataset(args.image_folder, args.anno_folder, subset_size=args.data_size, seed=args.seed)
    sampler = DistributedSampler(dataset, num_replicas=args.num_chunks, rank=args.chunk_idx, seed=args.seed, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, collate_fn=custom_collate_fn, drop_last=False)

    output_file = os.path.expanduser(args.output_file)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, "a") as out_file:
        for image, image_names, image_path in tqdm(dataloader):
            qs = DEFAULT_IMAGE_TOKEN + "\nPlease describe this image in detail."
            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)
            input_ids = input_ids.repeat(len(image), 1)
            image_tensor = process_images(image, image_processor, model.config)
            editor.edit_strength = args.edit_strength

            global CURRENT_STEP_ACTION
            CURRENT_STEP_ACTION = None

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.half().to(device),
                    do_sample=False,
                    max_new_tokens=512,
                    use_cache=True
                )

            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

            results = []
            for image_name, text in zip(image_names, outputs):
                img_id = int(image_name.split('_')[-1].split('.')[0])
                results.append(json.dumps({
                    "image_id": img_id,
                    "caption": text.split('ASSISTANT:')[-1].strip()
                }, ensure_ascii=False))

            with FileLock(f"000.lock"):
                out_file.write('\n'.join(results) + '\n')
                out_file.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--editor-path", type=str, required=True)
    parser.add_argument("--router-path", type=str, default=None, help="Path to trained DPO Router")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--image-folder", type=str, required=True)
    parser.add_argument("--anno-folder", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-size", type=int, default=2000)
    parser.add_argument("--output-file", type=str, default="output/gen_cap.json")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--edit-strength", type=float, default=1.0)
    args = parser.parse_args()

    eval_model(args)