#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cpu"
        # for multi-scale viewing reconstruction
        self.lod = 30
        self.llffhold = 8
        self.eval = True
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class SceneExpensionParams(ParamGroup):
    def __init__(self, parser):
        # general scene expension configs
        self.do_expension = True
        self.multiscale_datasets = True
        self.consecutive_expension = False
        self.expension_start_iter = 21000
        self.scale_blur_img = False
        self.upscale_blur_end_iter = 16000
        self.expension_end_iter = 29000
        self.expension_iter_list = [6000, 9000, 12000, 15000, 18000, 21000, 24000, 27000, 29000]
        self.expension_interval = 1000
        self.use_random_variant = True
        self.random_noise_scales = [0.1, 0.2]
        self.expen_one_iter_only = False
        self.scene_variant_num = 2
        self.loss_weight = [0.15, 0.05]
        self.gen_variant_path = "./variants_output/"
        self.img2img_config_path = "./sd_config/v2-inference-v.yaml"
        self.img2img_model_path = "./sd_ckpt/v2-1_768-ema-pruned.ckpt"
        self.sd_device = "cuda:1"
        self.img2img_batch_size = 1
        self.img2img_prompt = "professional graph with sharp and natural detail"
        # img2img ddim step is img2img_ddim_steps*broken_strength
        self.img2img_ddim_steps = 100
        self.broken_strength = [0.03, 0.01]
        self.img2img_ddim_eta = 0.3
        # img2img prompt importance scale
        self.img2img_scale = 5
        self.scaled_multi = 3
        self.save_img2img_images = True
        # scale and blur image config
        self.upscale_config_path = "./sd_config/x4-upscaling.yaml"
        self.upscale_model_path = "./sd_ckpt/x4-upscaler-ema.ckpt"
        self.upscale_num_sample = 1
        self.upscale_noise_level = [60, 30]
        self.upscale_ddim_steps = 30
        self.upscale_ddim_scale = 5
        self.upscale_ddim_eta = 0.3
        self.upscale_ddim_prompt = "professional graph with sharp and natural detail"
        self.upscale_ddim_seed = 42
        self.save_upscale_images = True
        super().__init__(parser, "Scene Expension Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.random_background = False
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
