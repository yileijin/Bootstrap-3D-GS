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
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

import numpy as np
from scene.cameras import Camera
from scene.colmap_loader import qvec2rotmat
from utils.general_utils import PILtoTorch

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        self.expended_cams = None

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.depths, args.eval, args.train_test_exp)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.depths, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        self.scene_info = scene_info
        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, False)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, True)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"), args.train_test_exp)
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, scene_info.train_cameras, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        exposure_dict = {
            image_name: self.gaussians.get_exposure_from_name(image_name).detach().cpu().numpy().tolist()
            for image_name in self.gaussians.exposure_mapping
        }

        with open(os.path.join(self.model_path, "exposure.json"), "w") as f:
            json.dump(exposure_dict, f, indent=2)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    def expend_camera_variants(self, opt):
        camera_list = self.getTrainCameras()
        if opt.use_random_variant:
            self.expended_cams = expend_random_cam(opt, camera_list)
        else:
            self.expended_cams = expend_consecutive_cam(opt, camera_list)
        self.expended_cam_num = len(self.expended_cams)

    def convert_expended_cams(self, denoised_imgs):
        # Convert the expanded cameras into the standard format
        for i in range(len(denoised_imgs)):
            org_cam = self.expended_cams[i]
            # img_tensor = PILtoTorch(denoised_imgs[i], denoised_imgs[i].size)[:3, ...]
            reimged_cam = Camera(resolution=org_cam.resolution, colmap_id=org_cam.colmap_id, 
                  R=org_cam.R, T=org_cam.T, FoVx=org_cam.FoVx, FoVy=org_cam.FoVy, 
                  depth_params=org_cam.depth_params, image=denoised_imgs[i], invdepthmap=org_cam.invdepthmap, 
                  image_name=org_cam.image_name, uid=org_cam.uid, data_device=org_cam.data_device,
                  train_test_exp=org_cam.train_test_exp, is_test_dataset=org_cam.is_test_dataset,
                  is_test_view=org_cam.is_test_view)
            self.expended_cams[i] = reimged_cam


def construct_random_R_and_T(camera, qscale=0.1, tscale=0.2, multi_scale=False):
    # For Multi_scale Random Construction, T is very sensitive. So we do not operate T values.
    if not multi_scale:
        new_R = camera.R + np.random.uniform(low=-1, high=1, size=camera.R.shape)* qscale
        new_R = new_R / np.sqrt(np.sum(new_R* new_R, axis=1))
        new_T = camera.T + np.random.uniform(low=-1, high=1, size=camera.T.shape)* tscale
    else:
        new_R = camera.R + np.random.uniform(low=-1, high=1, size=camera.R.shape)* qscale
        new_R = new_R / np.sqrt(np.sum(new_R* new_R, axis=1))
        new_T = camera.T
    return new_R, new_T

def construct_consecutive_R_and_T(camera1, camera2, variant_num=1, multi_scale=False):
    Rs, Ts = [], []
    for i in range(variant_num):
        multi = (i+1) / (variant_num + 1)
        new_R = camera1.R + multi * (camera2.R - camera1.R)
        new_R = new_R / np.sqrt(np.sum(new_R*new_R, axis=1))
        new_T = camera1.T + multi * (camera2.T - camera1.T)
        Rs.append(new_R)
        Ts.append(new_T)
    return Rs, Ts

def expend_random_cam(opt, train_cameras_list):
    """
    expend original loaded camera list with their variants
    """
    org_len = len(train_cameras_list)
    new_cam_list = []
    variant_num = opt.scene_variant_num
    noise_scale = opt.random_noise_scales
    multi_scale = opt.multi_scale
    for i in range(org_len):
        org_cam = train_cameras_list[i]
        
        for j in range(variant_num):
            new_R, new_T = construct_random_R_and_T(train_cameras_list[i],  noise_scale[0], noise_scale[1], multi_scale)  
            new_cam = Camera(resolution=org_cam.resolution, colmap_id=org_cam.colmap_id, R=new_R, T=new_T, 
                  FoVx=org_cam.FoVx, FoVy=org_cam.FoVy, depth_params=org_cam.depth_params,
                  image=org_cam.original_image, invdepthmap=org_cam.invdepthmap, 
                  image_name=org_cam.image_name, uid=org_cam.uid, data_device=org_cam.data_device,
                  train_test_exp=org_cam.train_test_exp, is_test_dataset=org_cam.is_test_dataset,
                  is_test_view=org_cam.is_test_view)
            new_cam_list.append(new_cam)
    return new_cam_list

def expend_consecutive_cam(opt, train_cameras_list):
    org_len = len(train_cameras_list)
    variant_num = opt.scene_variant_num
    multi_scale = opt.multi_scale
    new_cam_list = []
    for i in range(org_len-1):
        org_cam = train_cameras_list[i]
        Rs, Ts = construct_consecutive_R_and_T(train_cameras_list[i], train_cameras_list[i+1], variant_num, multi_scale)
        for j in range(variant_num):
            new_R, new_T = Rs[j], Ts[j]
            new_cam = Camera(resolution=org_cam.resolution, colmap_id=org_cam.colmap_id, R=new_R, T=new_T, 
                  FoVx=org_cam.FoVx, FoVy=org_cam.FoVy, depth_params=org_cam.depth_params,
                  image=org_cam.original_image, invdepthmap=org_cam.invdepthmap, 
                  image_name=org_cam.image_name, uid=org_cam.uid, data_device=org_cam.data_device,
                  train_test_exp=org_cam.train_test_exp, is_test_dataset=org_cam.is_test_dataset,
                  is_test_view=org_cam.is_test_view)
            new_cam_list.append(new_cam)
    return new_cam_list

def save_expended_cam(expended_cams, save_path):
    expended_cam_list = []
    for cam in expended_cams:
        cam_dict = {"uid":cam.uid, "R":cam.R.tolist(), "T":cam.T.tolist(), "FovY":cam.FovY,
                    "FovX":cam.FovX, "image": np.array(cam.image).tolist(), "image_path":cam.image_path,
                    "image_name":cam.image_name, "width":cam.width, "height":cam.height}
        expended_cam_list.append(cam_dict)
    with open(save_path, 'w') as df:
        for cam_dict in expended_cam_list:
            line = json.dumps(cam_dict)
            df.write(line)
            df.write("\n")
