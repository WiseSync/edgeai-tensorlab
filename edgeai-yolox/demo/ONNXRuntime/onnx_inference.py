#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import cv2
import numpy as np
import onnxruntime as rt

from yolox.data.data_augment import preproc as preprocess
from yolox.data.datasets import COCO_CLASSES
from yolox.utils import mkdir, multiclass_nms, demo_postprocess, vis


def make_parser():
    parser = argparse.ArgumentParser("onnxruntime inference sample")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="yolox.onnx",
        help="Input your onnx model.",
    )
    parser.add_argument(
        "-i",
        "--image_path",
        type=str,
        default='test_image.png',
        help="Path to your input image.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default='demo_output',
        help="Path to your output directory.",
    )
    parser.add_argument(
        "-s",
        "--score_thr",
        type=float,
        default=0.3,
        help="Score threshould to filter the result.",
    )
    parser.add_argument(
        "--input_shape",
        type=str,
        default="640,640",
        help="Specify an input shape for inference.",
    )
    parser.add_argument(
        "--with_p6",
        action="store_true",
        help="Whether your model uses p6 in FPN/PAN.",
    )
    parser.add_argument(
        "--save-txt",
        action="store_true",
        help="Whether your model uses p6 in FPN/PAN.",
    )
    parser.add_argument("--export-det",  action='store_true', help='export the nms part in ONNX model')

    parser.add_argument(
        "--tidl-delegate",
         action="store_true",
         help="use tidl_delegate"
    )
    parser.add_argument(
        "--compile",
         action="store_true",
         help="use tidl_delegate and compile"
    )

    return parser


if __name__ == '__main__':
    args = make_parser().parse_args()

    input_shape = tuple(map(int, args.input_shape.split(',')))
    origin_img = cv2.imread(args.image_path)
    img, ratio = preprocess(origin_img, input_shape)
    prototxt = args.model.replace("onnx", "prototxt")
    assert os.path.exists(prototxt), "Prototxt not available. Please provide a prototxt {}".format(prototxt)

    if args.tidl_delegate:
        compile_options = {
            "artifacts_folder": "/workspace/work/edgeai-tensorlab/edgeai-yolox/models/artifacts",
            "tensor_bits": 8,
            "accuracy_level": 1,
            "debug_level": 3,
            "advanced_options:calibration_frames": 25,
            "advanced_options:calibration_iterations": 2,
            # "advanced_options:output_feature_16bit_names_list" : "370, 680, 990, 1300",
            'object_detection:meta_layers_names_list': prototxt,
            'object_detection:meta_arch_type': 6
        }
        if args.compile:
            """copy_path = args.model[:-5] + "_opt.onnx"
            optimize(args.model, copy_path)
            args.model = copy_path """
            EP_list = ['TIDLCompilationProvider','CPUExecutionProvider']
            compile_options["tidl_tools_path"] = os.environ["TIDL_TOOLS_PATH"]
            os.makedirs(compile_options["artifacts_folder"], exist_ok=True)
            for root, dirs, files in os.walk(compile_options["artifacts_folder"], topdown=False):
                [os.remove(os.path.join(root, f)) for f in files]
                [os.rmdir(os.path.join(root, d)) for d in dirs]
        else:
            EP_list = ['TIDLExecutionProvider','CPUExecutionProvider']
            compile_options["tidl_tools_path"] = ""
        session = rt.InferenceSession(args.model ,providers=EP_list, provider_options=[compile_options, {}], sess_options=so)
    else:
        compile_options = {}
        EP_list = ['CPUExecutionProvider']
        session = rt.InferenceSession(args.model ,providers=EP_list, provider_options=[compile_options], sess_options=so)

    ort_inputs = {session.get_inputs()[0].name: img[None, :, :, :]}
    output = session.run(None, ort_inputs)
    if not args.export_det:
        predictions = demo_postprocess(output[0], input_shape, p6=args.with_p6)[0]

        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]

        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
        boxes_xyxy /= ratio
        dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)
    else:
        dets = output[0]
        dets[:, :4] /= ratio
    if dets is not None:
        final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
        origin_img = vis(origin_img, final_boxes, final_scores, final_cls_inds,
                         conf=args.score_thr, class_names=COCO_CLASSES)

    mkdir(args.output_dir)
    output_path = os.path.join(args.output_dir, args.image_path.split("/")[-1])
    cv2.imwrite(output_path, origin_img)


    if args.save_txt:  # Write to file in tidl dump format
        output_txt_path = os.path.join(os.path.dirname(output_path) , os.path.basename(output_path).split('.')[0] + '.txt')
        for *xyxy, conf, cls in dets.tolist():
            line = (conf, cls, *xyxy)
            if conf>args.score_thr:
                with open(output_txt_path, 'a') as f:
                    f.write(('%g ' * len(line)).rstrip() % line + '\n')