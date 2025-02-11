from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image
import torch
import os
import numpy as np
from PIL import Image
from random import randint

def tensor2PILimage(image_tensor):
    image = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    image = (image * 255).astype(np.uint8)
    image = Image.fromarray(image)
    return image

def reshape_image2up(image_list):
    new_image_list = []
    or_shapes = []
    for image in image_list:
        h, w = image.size
        or_shapes.append((h, w))
        a = 32
        if h % a != 0:
            h += a - h % a
        if w % a != 0:
            w += a - w % a
        image = image.resize((h, w))
        new_image_list.append(image)
    return new_image_list, or_shapes

def linear_ones_step_scheduled_strength(cur_iter, args):
    whole_span = args.exp_end_iter - args.exp_start_iter
    remained_per = (args.exp_end_iter - cur_iter) / whole_span
    cur_strength = args.broken_strength[0] - (1 - remained_per) * (args.broken_strength[0] - args.broken_strength[1])
    if args.infer_one_step:
        infer_step = int(1 / cur_strength + 1)
    else:
        infer_step = args.infer_steps
    return cur_strength, infer_step


def gen_image_variants(args, image_list, iteration):
    # The LoRA finetuning may not support variant=fp16, change it on your own , variant="fp16"
    pipe = AutoPipelineForImage2Image.from_pretrained(args.img2img_model_path, torch_dtype=torch.float16 , use_safetensors=True)#
    pipe.to(args.sd_device)
    if args.img2img_lora_path != "":
        pipe.load_lora_weights(args.img2img_lora_path)
    strength, infer_one_step = linear_ones_step_scheduled_strength(iteration, args)
    regenerated_imgs = []
    all_len = len(image_list)
    for i in range(0, all_len, args.boot_batch):
        end = min(all_len, i+args.boot_batch)
        images = image_list[i:end]
        prompts = [args.img2img_prompt] * len(images)
        new_images = pipe(prompts, image=images, num_inference_steps=infer_one_step, 
                        strength=strength, guidance_scale=args.guidance_scale,
                        eta=args.eta).images
        for new_img in new_images:
            regenerated_imgs.append(new_img)

    if args.save_img2img_images:
        iteration = iteration if iteration is not None else 0
        save_dir = os.path.join(args.gen_variant_path, "/img2img", f'{iteration}')
        os.makedirs(save_dir, exist_ok=True)
        for i in range(len(regenerated_imgs)):
            regenerated_imgs[i].save(os.path.join(save_dir, f"{i}.png"))
    return regenerated_imgs
