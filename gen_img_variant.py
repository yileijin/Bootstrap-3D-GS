import torch
import numpy as np
from omegaconf import OmegaConf
from einops import rearrange, repeat
from torch import autocast
from tqdm import tqdm, trange
import cv2 as cv
import PIL
from PIL import Image, ImageFilter
from pytorch_lightning import seed_everything
import random
import copy
import os

from ldm.util import exists, instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.ddpm import LatentUpscaleDiffusion, LatentUpscaleFinetuneDiffusion

def load_model_from_config(config, ckpt, verbose=False):
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model

# Directly load for conveniance or load it in the function
# config = OmegaConf.load("")
# model = load_model_from_config(config, "")


def gen_sd_variants(sep, iteration, new_scene_renders, cam):
    sep.img2img_broken_strength = calculate_linear_decrease_broken_strength(sep, iteration)
    print(f"Scene Expension with broken strength {sep.img2img_broken_strength}")
    denoised_imgs = denoise_scene_variants(new_scene_renders, sep)
    if sep.save_img2img_images:
        for i in range(len(denoised_imgs)):
            os.makedirs(sep.gen_variant_path+ f'/{iteration}'+ '/img2img', exist_ok=True)
            save_path = os.path.join(sep.gen_variant_path, f'{iteration}', 'img2img', f"{i}.png")
            denoised_imgs[i].save(save_path)
    # blur and scale denoised images
    if sep.scale_blur_img and iteration <= sep.upscale_blur_end_iter:
        sep.current_upscale_noise_level = calculate_linear_decrease_noise_level(sep, iteration)
        rescaled_and_blurred_imgs = rescale_and_blur_image(denoised_imgs, sep)
        upscaled_imgs = upscale_imgs(rescaled_and_blurred_imgs, sep)
        
        #print(cam.original_image.shape)
        down_scaled_imgs = rescale_imgs(upscaled_imgs, cam.image_width, cam.image_height)
        if sep.save_upscale_images:
            for i in range(len(down_scaled_imgs)):
                os.makedirs(sep.gen_variant_path+ f'/{iteration}'+ '/upscale', exist_ok=True)
                save_path = os.path.join(sep.gen_variant_path, f'{iteration}', 'upscale', f"{i}.png")
                down_scaled_imgs[i].save(save_path)
        return denoised_imgs, down_scaled_imgs
    return denoised_imgs

def load_img(path):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    #print(f"loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 64
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    #print(image.shape)
    return 2. * image - 1.

@torch.no_grad()
def denoise_scene_variants(imgs_list, args):#, model=model
    # imgs_list: list of torch.tensor images
    # return list of PIL.Image images
    config = OmegaConf.load(f"{args.img2img_config_path}")
    model = load_model_from_config(config, f"{args.img2img_model_path}")
    device = torch.device(args.sd_device)
    model = model.to(device)
    sampler = DDIMSampler(model)
    # all the prompts of regeneration are the same
    batch_size = args.img2img_batch_size
    prompt = args.img2img_prompt
    prompts = batch_size* [prompt]
    variant_samples = []

    batch_inp = []
    # create batch inp for efficient memory use
    if batch_size > 1:
        for i in range(int(len(imgs_list)/batch_size)):
            batch = imgs_list[i*batch_size:(i+1)*batch_size]
            batch_inp.append(batch)
    else:
        batch_inp = imgs_list
    for i in range(len(batch_inp)):
        if batch_size>1:
            init_image = torch.concat([load_img(path) for path in batch_inp[i]], dim=0).to(device)
        else:
            init_image = load_img(batch_inp[i]).to(device)
        init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space
        sampler.make_schedule(ddim_num_steps=args.img2img_ddim_steps, ddim_eta=args.img2img_ddim_eta, verbose=False)
        t_enc = int(args.img2img_broken_strength * args.img2img_ddim_steps)
        precision_scope = autocast
        with precision_scope("cuda"):
            with model.ema_scope():
                uc = None
                if args.img2img_scale != 1.0:
                    model.cond_stage_model.to(device)
                    model.cond_stage_model.device = device
                    uc = model.get_learned_conditioning(batch_size * [""])
                if isinstance(prompts, tuple):
                    prompts = list(prompts)
                c = model.get_learned_conditioning(prompts)

                # encode (scaled latent)
                z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc] * batch_size).to(device))
                # decode it
                samples = sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=args.img2img_scale,
                                            unconditional_conditioning=uc, )
                x_samples = model.decode_first_stage(samples)
                x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
                # in convience for the subsequent resize and blur
                for x_sample in x_samples:
                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                    img = Image.fromarray(x_sample.astype(np.uint8))
                    variant_samples.append(img)
    return variant_samples

def rescale_and_blur_image(imgs_list, args):
    # imgs_list: list of PIL.Image images
    rescaled_imgs = []
    for i in range(len(imgs_list)):
        img = imgs_list[i]
        img_blurred_and_resized = img.filter(ImageFilter.GaussianBlur(3)).resize(
            (int(img.size[0]/args.scaled_multi), int(img.size[1]/args.scaled_multi)))
        rescaled_imgs.append(img_blurred_and_resized)
    return rescaled_imgs


def upscale_imgs(imgs_list, args):
    config = OmegaConf.load(args.upscale_config_path)
    model = instantiate_from_config(config.model)
    model.load_state_dict(torch.load(args.upscale_model_path)["state_dict"], strict=False)

    device = torch.device(args.sd_device)
    model = model.to(device)
    sampler = DDIMSampler(model)
    sampler.device = device
    upscaled_imgs = []
    for i in range(len(imgs_list)):
        init_image = imgs_list[i].convert("RGB")
        image, pad_w, pad_h = pad_image(init_image)  # resize to integer multiple of 32
        width, height = image.size
        noise_level = torch.Tensor(
            args.upscale_num_sample * [args.current_upscale_noise_level]).to(device).long()
        sampler.make_schedule(args.upscale_ddim_steps, ddim_eta=args.upscale_ddim_eta, verbose=False)
        result = paint(
            sampler=sampler,
            image=image,
            prompt=args.upscale_ddim_prompt,
            seed=args.upscale_ddim_seed,
            scale=args.upscale_ddim_scale,
            h=height, w=width, steps=args.upscale_ddim_steps,
            num_samples=args.upscale_num_sample,
            callback=None,
            noise_level=noise_level,
            device=device
        )
        padded_img = result[0].astype(np.uint8)
        img = Image.fromarray(padded_img[:-pad_h*4,:-pad_w*4,:])
        upscaled_imgs.append(img)
    return upscaled_imgs

def rescale_imgs(imgs_list, w, h):
    rescaled_imgs = []
    for i in range(len(imgs_list)):
        img = imgs_list[i]
        rescaled_img = img.resize((w, h))
        rescaled_imgs.append(rescaled_img)
    return rescaled_imgs

def pad_image(input_image):
    pad_w, pad_h = np.max(((2, 2), np.ceil(
        np.array(input_image.size) / 64).astype(int)), axis=0) * 64 - input_image.size
    im_padded = Image.fromarray(
        np.pad(np.array(input_image), ((0, pad_h), (0, pad_w), (0, 0)), mode='edge'))
    return im_padded, pad_w, pad_h

def paint(sampler, image, prompt, seed, scale, h, w, steps, num_samples=1, 
            callback=None, eta=0., noise_level=None, device=None):
    if device is not None:
        device = device
    else:
        device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = sampler.model
    seed_everything(seed)
    prng = np.random.RandomState(seed)
    start_code = prng.randn(num_samples, model.channels, h, w)
    start_code = torch.from_numpy(start_code).to(
        device=device, dtype=torch.float32)

    with torch.no_grad(),\
            torch.autocast("cuda"):
        batch = make_batch_sd(
            image, txt=prompt, device=device, num_samples=num_samples)
        model.cond_stage_model.to(device)
        model.cond_stage_model.device = device
        c = model.cond_stage_model.encode(batch["txt"])
        c_cat = list()
        if isinstance(model, LatentUpscaleFinetuneDiffusion):
            for ck in model.concat_keys:
                cc = batch[ck]
                if exists(model.reshuffle_patch_size):
                    assert isinstance(model.reshuffle_patch_size, int)
                    cc = rearrange(cc, 'b c (p1 h) (p2 w) -> b (p1 p2 c) h w',
                                   p1=model.reshuffle_patch_size, p2=model.reshuffle_patch_size)
                c_cat.append(cc)
            c_cat = torch.cat(c_cat, dim=1)
            # cond
            cond = {"c_concat": [c_cat], "c_crossattn": [c]}
            # uncond cond
            uc_cross = model.get_unconditional_conditioning(num_samples, "")
            uc_full = {"c_concat": [c_cat], "c_crossattn": [uc_cross]}
        elif isinstance(model, LatentUpscaleDiffusion):
            x_augment, noise_level = make_noise_augmentation(
                model, batch, noise_level)
            cond = {"c_concat": [x_augment],
                    "c_crossattn": [c], "c_adm": noise_level}
            # uncond cond
            uc_cross = model.get_unconditional_conditioning(num_samples, "")
            uc_full = {"c_concat": [x_augment], "c_crossattn": [
                uc_cross], "c_adm": noise_level}
        else:
            raise NotImplementedError()

        shape = [model.channels, h, w]
        samples, intermediates = sampler.sample(
            steps,
            num_samples,
            shape,
            cond,
            verbose=False,
            eta=eta,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=uc_full,
            x_T=start_code,
            callback=callback
        )
    with torch.no_grad():
        x_samples_ddim = model.decode_first_stage(samples)
    result = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
    result = result.cpu().numpy().transpose(0, 2, 3, 1) * 255
    return result

def make_batch_sd(
        image,
        txt,
        device,
        num_samples=1,
):
    image = np.array(image.convert("RGB"))
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0
    batch = {
        "lr": rearrange(image, 'h w c -> 1 c h w'),
        "txt": num_samples * [txt],
    }
    batch["lr"] = repeat(batch["lr"].to(device=device),
                         "1 ... -> n ...", n=num_samples)
    return batch


def make_noise_augmentation(model, batch, noise_level=None):
    x_low = batch[model.low_scale_key]
    x_low = x_low.to(memory_format=torch.contiguous_format).float()
    x_aug, noise_level = model.low_scale_model(x_low, noise_level)
    return x_aug, noise_level


def calculate_linear_decrease_broken_strength(sep, iteration):
    strength_span = sep.broken_strength[0] - sep.broken_strength[1]
    iter_span = sep.expension_end_iter - sep.expension_start_iter
    augment = strength_span / iter_span * (iter_span - iteration + sep.expension_start_iter)
    return sep.broken_strength[1] + augment

def calculate_linear_decrease_noise_level(sep, iteration):
    noise_span = sep.upscale_noise_level[0] - sep.upscale_noise_level[1]
    iter_span = sep.expension_end_iter - sep.expension_start_iter
    augment = noise_span / iter_span * (iter_span - iteration + sep.expension_start_iter)
    return sep.upscale_noise_level[1] + augment