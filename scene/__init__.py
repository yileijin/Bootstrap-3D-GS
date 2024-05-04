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
import copy
import numpy as np

from scene.cameras import Camera
from scene.colmap_loader import qvec2rotmat
from .dataset_readers import CameraInfo, SceneInfo
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from utils.general_utils import PILtoTorch

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, 
                 shuffle=True, resolution_scales=[1.0]):
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

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, args.lod, args.llffhold)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
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
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print(f"Training Cameras length {len(self.train_cameras[resolution_scale])}")
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
            print(f"Testing Cameras length {len(self.test_cameras[resolution_scale])}")

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    
    def expend_camera_variants(self, args, random_variant=False, variant_num=2, 
        scale=1.0, random_noise_scales=[0.1,0.2]):
        # images are fake images, where expended cameras only hold their original image
        # of the current variant
        camera_list = self.scene_info.train_cameras
        if random_variant:
            expended_cams = expend_random_cam(camera_list, variant_num, random_noise_scales)
        else:
            expended_cams = expend_consecutive_cam(camera_list, variant_num)
        self.expended_cams = cameraList_from_camInfos(expended_cams, scale, args)
        self.expended_cam_num = len(self.expended_cams)

    def expend_camera_variants_multiscale(self, args, variant_num=2, scale=1.0):
        # images are fake images, where expended cameras only hold their original image
        # of the current variant
        camera_list = self.scene_info.test_cameras
        expended_cams = expend_consecutive_cam_multiscale(camera_list, variant_num)
        self.expended_cams = cameraList_from_camInfos(expended_cams, scale, args)
        # total number = test_cam_num * (1 + variant_num)
        self.expended_cam_num = len(self.expended_cams)

    def reconstruct_expended_camera_variants(self, denoised_imgs, resolution, down_scaled_imgs=None):
        # assert len(denoised_imgs) == len(self.expended_cams) == self.expended_cam_num
        print("Reconstruct expended camera variants")
        for i in range(len(denoised_imgs)):
            org_cam = self.expended_cams[i]
            img_tensor = PILtoTorch(denoised_imgs[i], resolution)[:3, ...]
            reimged_cam = Camera(colmap_id=org_cam.colmap_id, R=org_cam.R, T=org_cam.T, 
                  FoVx=org_cam.FoVx, FoVy=org_cam.FoVy, 
                  image=img_tensor, gt_alpha_mask=None,
                  image_name=org_cam.image_name, uid=org_cam.uid, data_device=org_cam.data_device)
            self.expended_cams[i] = reimged_cam
            
        if down_scaled_imgs is not None:
            self.expended_cams_re_scaled = copy.deepcopy(self.expended_cams)
            for i in range(self.expended_cam_num):
                org_cam = self.expended_cams_re_scaled[i]
                img_tensor = PILtoTorch(down_scaled_imgs[i], resolution)[:3, ...]
                reimged_cam = Camera(colmap_id=org_cam.colmap_id, R=org_cam.R, T=org_cam.T, 
                    FoVx=org_cam.FoVx, FoVy=org_cam.FoVy, 
                    image=img_tensor, gt_alpha_mask=None,
                    image_name=org_cam.image_name, uid=org_cam.uid, data_device=org_cam.data_device)
                self.expended_cams_re_scaled[i] = reimged_cam
        #print(self.expended_cams[0].original_image.shape)

def construct_random_R_and_T(camera, qscale=0.1, tscale=0.2):
    qvec = camera.qvec + np.random.uniform(low=-1, high=1, size=camera.qvec.shape)*0.1
    qvec = qvec / np.sqrt(np.sum(qvec*qvec))
    R = np.transpose(qvec2rotmat(qvec))
    tvec = camera.tvec + np.random.uniform(low=-1, high=1, size=camera.tvec.shape)*0.2
    T = np.array(tvec)
    return R, T, qvec, tvec

# expend original loaded cameras with 4 times
def expend_random_cam(train_cameras_list, variant_num=1, noise_scale=[0.1, 0.2]):
    """
    expend original loaded camera list with their variants
    """
    org_len = len(train_cameras_list)
    new_cam_list = []
    for i in range(org_len):
        for j in range(variant_num):
            new_uid = org_len + len(new_cam_list) + 1
            new_R, new_T, new_qvec, new_tvec = construct_random_R_and_T(train_cameras_list[i], 
                    noise_scale[0], noise_scale[1])
            new_image_name = train_cameras_list[i].image_name + f'variant_{i}_{j}'
            new_cam = CameraInfo(uid=new_uid, R=new_R, T=new_T, qvec=new_qvec, tvec=new_tvec, 
            FovY=train_cameras_list[i].FovY, FovX=train_cameras_list[i].FovX, 
            image=train_cameras_list[i].image, image_path=train_cameras_list[i].image_path, 
            image_name=new_image_name, width=train_cameras_list[i].width, 
            height=train_cameras_list[i].height)
            new_cam_list.append(new_cam)
    return new_cam_list


def construct_consecutive_R_and_T(camera1, camera2, variant_num=1):
    qvecs, tvecs, Rs, Ts = [], [], [], []
    for i in range(variant_num):
        multi = (i+1) / (variant_num + 1)
        qvec = camera1.qvec + multi * (camera2.qvec - camera1.qvec)
        qvec = qvec / np.sqrt(np.sum(qvec*qvec))
        R = np.transpose(qvec2rotmat(qvec))
        tvec = camera1.tvec + multi * (camera2.tvec - camera1.tvec)
        T = np.array(tvec)
        qvecs.append(qvec)
        tvecs.append(tvec)
        Rs.append(R)
        Ts.append(T)
    return Rs, Ts, qvecs, tvecs

def expend_consecutive_cam(train_cameras_list, variant_num=1):
    """
    expend original loaded camera list with their variants
    """
    org_len = len(train_cameras_list)
    new_cam_list = []
    for i in range(org_len-1):
        Rs, Ts, qvecs, tvecs = construct_consecutive_R_and_T(train_cameras_list[i], 
                                train_cameras_list[i+1], variant_num)
        for j in range(variant_num):
            new_uid = org_len + len(new_cam_list) + 1
            new_R, new_T, new_qvec, new_tvec = Rs[j], Ts[j], qvecs[j], tvecs[j]
            new_image_name = train_cameras_list[i].image_name + f'variant_{i}_{j}'
            new_cam = CameraInfo(uid=new_uid, R=new_R, T=new_T, qvec=new_qvec, tvec=new_tvec, 
            FovY=train_cameras_list[i].FovY, FovX=train_cameras_list[i].FovX, 
            image=train_cameras_list[i].image, image_path=train_cameras_list[i].image_path, 
            image_name=new_image_name, width=train_cameras_list[i].width, 
            height=train_cameras_list[i].height)
            new_cam_list.append(new_cam)
    return new_cam_list

def expend_consecutive_cam_multiscale(train_cameras_list, variant_num=1):
    """
    expend test camera list with their variants
    """
    org_len = len(train_cameras_list)
    new_cam_list = train_cameras_list
    for i in range(org_len-1):
        Rs, Ts, qvecs, tvecs = construct_consecutive_R_and_T(train_cameras_list[i], 
                                train_cameras_list[i+1], variant_num)
        for j in range(variant_num):
            new_uid = org_len + len(new_cam_list) + 1
            new_R, new_T, new_qvec, new_tvec = Rs[j], Ts[j], qvecs[j], tvecs[j]
            new_image_name = train_cameras_list[i].image_name + f'variant_{i}_{j}'
            new_cam = CameraInfo(uid=new_uid, R=new_R, T=new_T, qvec=new_qvec, tvec=new_tvec, 
            FovY=train_cameras_list[i].FovY, FovX=train_cameras_list[i].FovX, 
            image=train_cameras_list[i].image, image_path=train_cameras_list[i].image_path, 
            image_name=new_image_name, width=train_cameras_list[i].width, 
            height=train_cameras_list[i].height)
            new_cam_list.append(new_cam)
    return new_cam_list

def save_expended_cam(expended_cams, save_path):
    expended_cam_list = []
    for cam in expended_cams:
        cam_dict = {"uid":cam.uid, "R":cam.R.tolist(), "T":cam.T.tolist(), "FovY":cam.FovY,
                    "FovX":cam.FovX, "image": np.array(cam.image).tolist(), "image_path":cam.image_path,
                    "image_name":cam.image_name, "width":cam.width, "height":cam.height,
                    "qvec":cam.qvec.tolist(), "tvec":cam.tvec.tolist()}
        expended_cam_list.append(cam_dict)
    with open(save_path, 'w') as df:
        for cam_dict in expended_cam_list:
            line = json.dumps(cam_dict)
            df.write(line)
            df.write("\n")