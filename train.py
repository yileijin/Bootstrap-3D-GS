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

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene.dataset_readers import CameraInfo
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from utils.camera_utils import cameraList_from_camInfos
import uuid
from tqdm import tqdm
import torchvision

from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, SceneExpensionParams
from gen_img_variant import gen_sd_variants
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, sep, testing_iterations, saving_iterations, 
             checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset, sep)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)

    if sep.do_expension:
        scene.expended_cams = None

    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        if sep.do_expension and (scene.expended_cams is not None) and sep.multiscale_datasets:
            pop_view_idx = randint(0, len(viewpoint_stack)+len(scene.expended_test_cams)-1)
            if pop_view_idx <= dataset.lod:
                viewpoint_cam = scene.expended_test_cams[pop_view_idx]
            else:
                viewpoint_cam = viewpoint_stack.pop(pop_view_idx-dataset.lod-1)
        else:
            pop_view_idx = randint(0, len(viewpoint_stack)-1)
            viewpoint_cam = viewpoint_stack.pop(pop_view_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        if sep.do_expension and (scene.expended_cams is not None) and (not sep.multiscale_datasets):
            # Take both two sides of rendered cams
            start = max((pop_view_idx-1)*sep.scene_variant_num, 0)
            end = min((pop_view_idx+1)* sep.scene_variant_num, len(scene.expended_cams))
            variant_cams = scene.expended_cams[start:end].copy()
            if sep.scale_blur_img and iteration <= sep.upscale_blur_end_iter and scene.expended_cams_re_scaled is not None:
                # We do this setting for memory issues
                # variant_cams = variant_cams[1:-1]
                variant_cams.extend(scene.expended_cams_re_scaled[pop_view_idx* sep.scene_variant_num:
                (pop_view_idx+1)* sep.scene_variant_num].copy())
            add_loss = torch.tensor(0.).cuda()
            exp_viewspace_point_tensors = []
            exp_visibility_filters = []
            exp_radiis = []

            for cam in variant_cams:
                # render and compute loss
                expended_image = cam.original_image.cuda()
                exp_render_pkg = render(cam, gaussians, pipe, bg)
                exp_image, exp_viewspace_point_tensor, exp_visibility_filter, exp_radii = exp_render_pkg["render"], exp_render_pkg["viewspace_points"], exp_render_pkg["visibility_filter"], exp_render_pkg["radii"]
                sub_Ll1 = l1_loss(exp_image, expended_image)
                #sub_loss = (1.0 - opt.lambda_dssim) * sub_Ll1 + opt.lambda_dssim * (1.0 - ssim(exp_image, expended_image))
                add_loss = add_loss + sub_Ll1
                # prepare for densification
                exp_viewspace_point_tensors.append(exp_viewspace_point_tensor)
                exp_visibility_filters.append(exp_visibility_filter)
                exp_radiis.append(exp_radii)
            # we scale up the bootstrapping loss in practice
            add_loss = add_loss / len(variant_cams) * 3

            if (iteration % sep.expension_interval) <= (sep.expension_interval / 2):
                expen_loss_weight = sep.loss_weight[0]
            else:
                expen_loss_weight = sep.loss_weight[1]
            loss = (1.0 - expen_loss_weight) * loss + expen_loss_weight * add_loss

        elif sep.do_expension and (scene.expended_cams is not None) and sep.multiscale_datasets:
            if pop_view_idx <= dataset.lod:
                # Take both two sides of rendered cams
                start = max((pop_view_idx)*sep.scene_variant_num, 0)
                end = min((pop_view_idx+1)* sep.scene_variant_num, len(scene.expended_cams))
                variant_cams = scene.expended_cams[start:end].copy()
                if sep.scale_blur_img and iteration <= sep.upscale_blur_end_iter and scene.expended_cams_re_scaled is not None:
                    # We do this setting for memory issues
                    # variant_cams = variant_cams[1:-1]
                    variant_cams.extend(scene.expended_cams_re_scaled[pop_view_idx* sep.scene_variant_num:
                    (pop_view_idx+1)* sep.scene_variant_num].copy())
                add_loss = torch.tensor(0.).cuda()
                exp_viewspace_point_tensors = []
                exp_visibility_filters = []
                exp_radiis = []

                for cam in variant_cams:
                    # render and compute loss
                    expended_image = cam.original_image.cuda()
                    exp_render_pkg = render(cam, gaussians, pipe, bg)
                    exp_image, exp_viewspace_point_tensor, exp_visibility_filter, exp_radii = exp_render_pkg["render"], exp_render_pkg["viewspace_points"], exp_render_pkg["visibility_filter"], exp_render_pkg["radii"]
                    sub_Ll1 = l1_loss(exp_image, expended_image)
                    #sub_loss = (1.0 - opt.lambda_dssim) * sub_Ll1 + opt.lambda_dssim * (1.0 - ssim(exp_image, expended_image))
                    add_loss = add_loss + sub_Ll1
                    # prepare for densification
                    exp_viewspace_point_tensors.append(exp_viewspace_point_tensor)
                    exp_visibility_filters.append(exp_visibility_filter)
                    exp_radiis.append(exp_radii)

                loss = (loss + add_loss) / (len(variant_cams) + 1) * 0.05

        loss.backward()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
            
            if sep.do_expension:
                do_expension = False
                if sep.consecutive_expension:
                    if (iteration >= sep.expension_start_iter-1) and (iteration % sep.expension_interval == sep.expension_start_iter % sep.expension_interval) and (iteration <= sep.expension_end_iter):
                        do_expension = True
                else:
                    if iteration in sep.expension_iter_list:
                        do_expension = True
                    elif sep.expen_one_iter_only and iteration not in sep.expension_iter_list and (iteration % sep.expension_interval == sep.expension_start_iter % sep.expension_interval):
                        scene.expended_cams = None
                        scene.expended_cams_re_scaled = None
                if do_expension:
                    if sep.multiscale_datasets:
                        #print("Expend for Multiscale Reconstruction Datasets")
                        scene.expend_camera_variants_multiscale(dataset, variant_num=sep.scene_variant_num)
                    else:
                        scene.expend_camera_variants(dataset, random_variant=sep.use_random_variant, variant_num=sep.scene_variant_num, 
                            random_noise_scales=sep.random_noise_scales)
                    new_scene_renders = []
                    for i in range(len(scene.expended_cams)):
                        rendered_img = render(scene.expended_cams[i], 
                                                        gaussians, pipe, background)["render"][0:3, :, :]
                        os.makedirs(sep.gen_variant_path+ f'/{iteration}', exist_ok=True)
                        save_path = os.path.join(sep.gen_variant_path, f'{iteration}', f"{i}.png")
                        torchvision.utils.save_image(rendered_img, save_path)
                        new_scene_renders.append(save_path)
                    cam = scene.getTrainCameras()[0]
                    #print(cam.image_width, cam.image_height)
                    if sep.scale_blur_img and iteration <= sep.upscale_blur_end_iter:
                        denoised_imgs, down_scaled_imgs = gen_sd_variants(sep, iteration, new_scene_renders, cam)
                        scene.reconstruct_expended_camera_variants(denoised_imgs, (cam.image_width, cam.image_height), down_scaled_imgs)
                    else:
                        denoised_imgs = gen_sd_variants(sep, iteration, new_scene_renders, cam)
                        scene.reconstruct_expended_camera_variants(denoised_imgs, (cam.image_width, cam.image_height))
                    
                    if sep.multiscale_datasets:
                        scale = 1.0
                        scene.expended_test_cams = scene.expended_cams[:dataset.lod+1]
                        scene.expended_cams = scene.expended_cams[dataset.lod+1:]
                        scene.expended_cam_num = len(scene.expended_cams)
                        #assert scene.expended_cam_num == (dataset.lod + 1) * sep.scene_variant_num
            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                if sep.do_expension and scene.expended_cams is not None and (iteration % sep.expension_interval != sep.expension_start_iter % sep.expension_interval):
                    if sep.multiscale_datasets and pop_view_idx > dataset.lod:
                        pass
                    else:
                        # Expension Prune
                        for i in range(len(exp_viewspace_point_tensors)):
                            gaussians.max_radii2D[exp_visibility_filters[i]] = torch.max(gaussians.max_radii2D[exp_visibility_filters[i]], 
                                                                                exp_radiis[i][exp_visibility_filters[i]])
                            gaussians.add_densification_stats(exp_viewspace_point_tensors[i], exp_visibility_filters[i])

                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args, sep):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    with open(os.path.join(args.model_path, "sep_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(sep))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    sep = SceneExpensionParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 10000, 13000, 16000, 19000, 22000, 25000, 28000, 30000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 10000, 13000, 16000, 19000, 22000, 25000, 28000, 30000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), sep,
             args.test_iterations, args.save_iterations, args.checkpoint_iterations, 
             args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
