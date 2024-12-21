#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os
from loguru import logger

import cv2
import numpy as np
from math import acos, sin, sqrt, copysign
from pycocotools.coco import COCO
from plyfile import PlyData
import yaml
import json

from ..dataloading import get_yolox_datadir
from .datasets_wrapper import Dataset

root_path = "/workspace/Datasets/nuScenes/v1.0-mini/"

def get_camera_matrix(calib):
    """
    從 calib (3×4) 中擷取相機內參 (3×3)。
    calib 一般形如：
    [
    [fx,  0, cx, Tx],
    [ 0, fy, cy, Ty],
    [ 0,  0,  1,  0]
    ]
    回傳 camera_matrix (3×3)。
    """
    # 將 calib 轉為 numpy array，方便切片
    calib_array = np.array(calib)
    # 取前 3 個欄位，即為 camera intrinsic matrix
    camera_matrix = calib_array[:, :3]
    return camera_matrix

def get_3d_box_corners_from_dim(dim):
    """
    根據 dim = [height, width, length]
    回傳 shape = (3, 8) 的 3D bounding box 8 個頂點坐標 (x, y, z)，
    物件中心位於 (0, 0, 0)。
    """
    h, w, l = dim[0]*1000, dim[1]*1000, dim[2]*1000
    
    # 以物件中心 (0,0,0) 為原點
    # x軸方向: ±(l/2), y軸方向: ±(h/2), z軸方向: ±(w/2)
    # （您可依照實際定義更改：像是把 y=0 設在底部）
    # x軸: ±l/2, y軸: [0 ~ -h], z軸: ±w/2
    x_corners = [ l/2,  l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2]
    y_corners = [ 0,     0,     0,     0,  -h,   -h,   -h,   -h ]
    z_corners = [ w/2, -w/2, -w/2,  w/2,  w/2, -w/2, -w/2,  w/2]
    
    # 組合為 (3, 8) 的 numpy array: [[x1..x8],[y1..y8],[z1..z8]]
    corners_3d = np.vstack((x_corners, y_corners, z_corners))
    return corners_3d

def get_bounding_box_diameter(corners_3d):
    """
    corners_3d: shape (8, 3)，
    回傳該 3D box 8個頂點之間的最長距離 (float)。
    """
    points = corners_3d  # shape => (8, 3)
    
    max_dist = 0.0
    num_pts = points.shape[0]
    # 兩兩比對
    for i in range(num_pts):
        for j in range(i+1, num_pts):
            dist = np.linalg.norm(points[i] - points[j])
            if dist > max_dist:
                max_dist = dist
    return max_dist

class CADModelsNuScenes():
    def __init__(self, camera_matrix,data_dir=None):
        if data_dir is None:
            data_dir = root_path
        self.data_dir = data_dir
        self.cad_models_path = os.path.join(self.data_dir, "models_eval")
        self.class_to_name = {
                    0: "car" , 1: "truck", 2: "trailer", 3: "bus", 4: "construction_vehicle", 5: "pedestrian", 6: "motorcycle", 7: "bicycle", 8: "traffic_cone", 9: "barrier"
                    }
        self.class_to_model = self.load_cad_models()
        self.class_to_sparse_model = self.create_sparse_models()
        self.models_corners, self.models_diameter = self.get_models_params()
        self.camera_matrix = camera_matrix
        self.symmetric_objects = { }

    def load_cad_models(self):
        dims = {0: np.array([1.52131309, 1.64441358, 3.85728004]),
                1: np.array([3.07044968,  2.62877944, 11.17126338]),
                2: np.array([2.18560847, 1.91077601, 5.08042328]),
                3: np.array([3.07044968,  2.62877944, 11.17126338]),
                4: np.array([1.52131309, 1.64441358, 3.85728004]),
                5: np.array([1.75562272, 0.67027992, 0.87397566]),
                6: np.array([1.73456498, 0.58174006, 1.77485499]),
                7: np.array([1.73456498, 0.58174006, 1.77485499]),
                8: np.array([1.75562272, 0.67027992, 0.87397566]),
                9: np.array([1.75562272, 0.67027992, 0.87397566])}
        class_to_model = {class_id: None for class_id in self.class_to_name.keys()}

        for class_id in self.class_to_name.keys():
            dim = dims[class_id]
            dim = [dim[0]*1000, dim[1]*1000, dim[2]*1000]
            model = get_3d_box_corners_from_dim(dims[class_id])
            model = model.T # shape => (8, 3)
            class_to_model[class_id] = model

        return class_to_model

    def get_models_params(self):
        """
        Convert model corners from (min_x, min_y, min_z, size_x, size_y, size_z) to actual coordinates format of dimension (8,3)
        Return the corner coordinates and the diameters of each models
        """
        models_corners_3d = {}
        models_diameter = {}
        for model_id, corners_3d in self.class_to_model.items():
            models_corners_3d.update({int(model_id): corners_3d})
            diameter = get_bounding_box_diameter(corners_3d)

            models_diameter.update({int(model_id): diameter})
        return models_corners_3d, models_diameter

    def create_sparse_models(self):
        class_to_sparse_model = self.class_to_model.copy()
        return class_to_sparse_model


class NuScenesDataset(Dataset):
    """
    YCBV dataset class.
    """

    def __init__(
        self,
        data_dir=None,
        json_file="mini_train.json",
        name="train",
        img_size=(480, 640),
        preproc=None,
        cache=False,
        object_pose=False,
        symmetric_objects=None
    ):
        """
        YCBV dataset initialization. Annotation data are read into memory by COCO API.
        Args:
            data_dir (str): dataset root directory
            json_file (str): YCBV json file name
            name (str): YCBV data name
            img_size (int): target image size after pre-processing
            preproc: data augmentation strategy
        """
        super().__init__(img_size)
        if data_dir is None:
            data_dir = root_path
        self.data_dir = data_dir
        self.json_file = json_file
        self.object_pose = object_pose 

        self.coco = COCO(os.path.join(self.data_dir, "annotations", self.json_file))
        self.ids = self.coco.getImgIds()
        self.class_ids = sorted(self.coco.getCatIds())
        cats = self.coco.loadCats(self.coco.getCatIds())
        self._classes = tuple([c["name"] for c in cats])
        self.imgs = None
        self.imgs_coco = self.coco.imgs
        self.name = name
        self.img_size = img_size
        img  = self.coco.imgs[self.ids[0]]
        camera_matrix = get_camera_matrix(img["calib"])
        camera_matrix = camera_matrix.reshape(9)
        self.cad_models = CADModelsNuScenes(camera_matrix=camera_matrix)
        self.models_corners, self.models_diameter = self.cad_models.models_corners, self.cad_models.models_diameter
        self.class_to_name = self.cad_models.class_to_name
        self.class_to_model = self.cad_models.class_to_model
        if preproc is not None:
            self.preproc = preproc
        self.annotations = self._load_coco_annotations()
        self.symmetric_objects = self.cad_models.symmetric_objects
        if cache:
            self._cache_images()
        self.detection = True

    def __len__(self):
        return len(self.ids)

    def __del__(self):
        del self.imgs

    def _load_coco_annotations(self):
        return [self.load_anno_from_ids(_ids) for _ids in self.ids]

    def _cache_images(self):
        logger.warning(
            "\n********************************************************************************\n"
            "You are using cached images in RAM to accelerate training.\n"
            "This requires large system RAM.\n"
            "Make sure you have 200G+ RAM and 136G available disk space for training COCO.\n"
            "********************************************************************************\n"
        )
        max_h = self.img_size[0]
        max_w = self.img_size[1]
        cache_file = self.data_dir + "/img_resized_cache_" + self.name + ".array"
        if not os.path.exists(cache_file):
            logger.info(
                "Caching images for the first time. This might take about 20 minutes for COCO"
            )
            self.imgs = np.memmap(
                cache_file,
                shape=(len(self.ids), max_h, max_w, 3),
                dtype=np.uint8,
                mode="w+",
            )
            from tqdm import tqdm
            from multiprocessing.pool import ThreadPool

            NUM_THREADs = min(8, os.cpu_count())
            loaded_images = ThreadPool(NUM_THREADs).imap(
                lambda x: self.load_resized_img(x),
                range(len(self.annotations)),
            )
            pbar = tqdm(enumerate(loaded_images), total=len(self.annotations))
            for k, out in pbar:
                self.imgs[k][: out.shape[0], : out.shape[1], :] = out.copy()
            self.imgs.flush()
            pbar.close()
        else:
            logger.warning(
                "You are using cached imgs! Make sure your dataset is not changed!!"
            )

        logger.info("Loading cached imgs...")
        self.imgs = np.memmap(
            cache_file,
            shape=(len(self.ids), max_h, max_w, 3),
            dtype=np.uint8,
            mode="r+",
        )

    def load_anno_from_ids(self, id_):
        im_ann = self.coco.loadImgs(id_)[0]
        width = im_ann["width"]
        height = im_ann["height"]
        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)
        annotations = self.coco.loadAnns(anno_ids)
        objs = []
        for obj in annotations:
            x1 = np.max((0, obj["bbox"][0]))
            y1 = np.max((0, obj["bbox"][1]))
            x2 = np.min((width, obj["bbox"][0] + np.max((0, obj["bbox"][2]))))
            y2 = np.min((height, obj["bbox"][1] + np.max((0, obj["bbox"][3]))))
            if obj["area"] > 0 and x2 >= x1 and y2 >= y1:
                obj["clean_bbox"] = [x1, y1, x2, y2]
                objs.append(obj)

        num_objs = len(objs)

        if self.object_pose:
            res = np.zeros((num_objs, 14))
        else:
            res = np.zeros((num_objs, 5))

        for ix, obj in enumerate(objs):
            cls = self.class_ids.index(obj["category_id"])
            res[ix, 0:4] = obj["clean_bbox"]
            res[ix, 4] = cls
            if self.object_pose:
                #image_folder  = int(im_ann['image_folder'])
                """if image_folder < 60:
                    camera_matrix = self.cad_models.camera_matrix['camera_uw']
                else:
                    camera_matrix = self.cad_models.camera_matrix['camera_cmu'] """
                camera_matrix = get_camera_matrix(im_ann["calib"])
                obj_centre_2d = np.matmul(camera_matrix.reshape(3,3), np.array(obj["T"])/obj["T"][2])[:2]  #rotation vec not required for the center point
                #res[ix, 11:14] = obj["T"]
                obj_centre_2d = np.squeeze(obj_centre_2d)
                res[ix, -3:-1] = obj_centre_2d
                res[ix, -1] = obj["T"][2] / 100.0
                #obj["R_aa"], _ = cv2.Rodrigues(np.array(obj["R"]).reshape(3,3))
                #obj["R_aa"] = np.squeeze(obj["R_aa"])
                #Use Gram-Schmidt to make the rotation representation continuous and in 6D
                #https://towardsdatascience.com/better-rotation-representations-for-accurate-pose-estimation-e890a7e1317f
                R_gs = np.array(obj["R"]).reshape(3,3)
                obj["R_gs"] = np.squeeze(R_gs[:, :2].transpose().reshape(6, 1))
                res[ix, -9:-3] = obj["R_gs"]
            #print(res[ix, 11:13])
        r = min(self.img_size[0] / height, self.img_size[1] / width)
        res[:, :4] *= r
        res[:, -3:-1] *= r

        img_info = (height, width)
        resized_info = (int(height * r), int(width * r))

        file_name = (
            im_ann["file_name"]
            if "file_name" in im_ann
            else "{:04}".format(id_) + ".png"
        )

        return (res, img_info, resized_info, file_name)

    def load_anno(self, index):
        return self.annotations[index][0]

    def load_resized_img(self, index):
        img = self.load_image(index)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        return resized_img

    def load_image(self, index):
        file_name = self.annotations[index][3]
        img_index = list(self.imgs_coco)[index]
        #image_folder = self.imgs_coco[int(img_index)]['image_folder']
        #type = self.imgs_coco[img_index]['type']
        img_file = os.path.join(self.data_dir, file_name)
        img = cv2.imread(img_file)
        assert img is not None

        return img

    def pull_item(self, index):
        id_ = self.ids[index]

        res, img_info, resized_info, _ = self.annotations[index]
        if self.imgs is not None:
            pad_img = self.imgs[index]
            img = pad_img[: resized_info[0], : resized_info[1], :].copy()
        else:
            img = self.load_resized_img(index)

        return img, res.copy(), img_info, index

    @Dataset.mosaic_getitem
    def __getitem__(self, index):
        """
        One image / label pair for the given index is picked up and pre-processed.

        Args:
            index (int): data index

        Returns:
            img (numpy.ndarray): pre-processed image
            padded_labels (torch.Tensor): pre-processed label data.
                The shape is :math:`[max_labels, 5]`.
                each label consists of [class, xc, yc, w, h]:
                    class (float): class index.
                    xc, yc (float) : center of bbox whose values range from 0 to 1.
                    w, h (float) : size of bbox whose values range from 0 to 1.
            info_img : tuple of h, w.
                h, w (int): original shape of the image
            img_id (int): same as the input index. Used for evaluation.
        """
        img, target, img_info, img_id = self.pull_item(index)

        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)
        return img, target, img_info, img_id
