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
import numpy as np
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from scene import expend_random_cam, expend_consecutive_cam
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, do_expension):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        if not do_expension:
            torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool,
    do_expension, expension_num_per_view, random_expension):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            views = scene.getTrainCameras()
            name = "train"
            if do_expension:
                if random_expension:
                    expended_cams = expend_random_cam(scene.scene_info.train_cameras, expension_num_per_view)
                else:
                    expended_cams = expend_consecutive_cam(scene.scene_info.train_cameras, expension_num_per_view)
                new_cams = cameraList_from_camInfos(expended_cams, 1.0, dataset)
                views.extend(new_cams)
                name = "train_boot"
            render_set(dataset.model_path, name, scene.loaded_iter, views, gaussians, pipeline, background, do_expension)

        if not skip_test:
            views = scene.getTestCameras()
            name = "test"
            if do_expension:
                if random_expension:
                    expended_cams = expend_random_cam(scene.scene_info.test_cameras, expension_num_per_view)
                else:
                    expended_cams = expend_consecutive_cam(scene.scene_info.test_cameras, expension_num_per_view)
                new_cams = cameraList_from_camInfos(expended_cams, 1.0, dataset)
                views.extend(new_cams)
                name = "test_boot"
            render_set(dataset.model_path, name, scene.loaded_iter, views, gaussians, pipeline, background, do_expension)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=[7000, 10000, 13000, 16000, 19000, 22000, 25000, 28000,30000], type=int)#7000, 10000, 13000, 16000, 19000, 22000, 25000, 28000,
    parser.add_argument("--skip_train", default=True)
    parser.add_argument("--do_expension", default=False)
    parser.add_argument("--expension_num_per_view", default=3)
    parser.add_argument("--random_expension", default=True)
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    
    # Initialize system state (RNG)
    safe_state(args.quiet)
    for iteration in args.iteration:
        render_sets(model.extract(args), iteration, pipeline.extract(args), args.skip_train, args.skip_test,
        args.do_expension, args.expension_num_per_view, args.random_expension)