# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import sys
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))
import asyncio
from argparse import ArgumentParser
from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)
import cv2
import numpy as np
from glob import glob
import json


def parse_args():
    parser = ArgumentParser()
    #parser.add_argument('img', help='Image file')
    parser.add_argument('--config', default='./configs/i3cl_vitae_fpn/i3cl_vitae_fpn_ms_train.py', help='Config file')
    parser.add_argument('--checkpoint', default='', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.45, help='bbox score threshold')
    parser.add_argument(
        '--async-test', default=False, action='store_true',
        help='whether to set async options for async inference.')
    parser.add_argument('--result_imgs_file', type=str, default='result_images/')
    parser.add_argument('--result_txt_file', type=str, default='result_txt/')
    parser.add_argument('--test_imgs_file', type=str, default='data/art/test_images/')
    parser.add_argument('--json_file', type=str, default='art_eval.json')
    args = parser.parse_args()
    return args


def format_art_output(txt_save_path, polygon_list, score_list):
    assert len(polygon_list) == len(score_list)
    f = open(txt_save_path, 'w')
    for i in range(len(polygon_list)):
        polygon = polygon_list[i]
        score = score_list[i]
        polygon = polygon.tolist()
        str_out = ''
        for p in polygon:
            str_out += str(p) + ','
        str_out += str(score) + ',' + '###'
        f.write(str_out)
        f.write('\n')
    f.close()


def main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    
    # test a single image
    os.makedirs(args.result_imgs_file, exist_ok=True)
    os.makedirs(args.result_txt_file, exist_ok=True)
    image_list = glob(os.path.join(args.test_imgs_file, '*'))
    total = len(image_list)
    count = 1
    
    for img_path in image_list:
        img_name = img_path.split('/')[-1]
        result = inference_detector(model, img_path)
        polygon_list, score_list = show_result_pyplot(model, img_path, result, score_thr=args.score_thr, out_file=os.path.join(args.result_imgs_file, img_name))
        format_art_output(
            os.path.join(args.result_txt_file, img_name.replace('.jpg', '.txt')),
            polygon_list,
            score_list
        )
        print(f'{img_name}, {count} / {total}', end='\r')
        count += 1
    
    """write json"""
    res_dict = {}
    txt_list = glob(os.path.join(args.result_txt_file, '*.txt'))
    total_len = len(txt_list)
    n = 1
    for k in txt_list:
        with open(k, 'r') as f:
            data = f.readlines()
        res_list = []
        for i in data:
            i = i.split(',')
            confidence = i[len(i)-2]
            points = i[0:len(i)-2]
            new_points = []
            for j in range(0, len(points), 2):
                new_points.append([round(float(points[j])), round(float(points[j+1]))])
                
            new_points = np.array(new_points)
            epsilon = 0.005 * cv2.arcLength(new_points, True)
            new_points2 = cv2.approxPolyDP(new_points, epsilon=epsilon, closed=True)
            new_points3 = []
            
            for m in new_points2:
                new_points3.append([int(float(m[0][0])), int(float(m[0][1]))])
            new_points3 = new_points3[::-1]
            if len(new_points3) < 3:continue

            res_list.append({"points": new_points3, "confidence": float(confidence)})
            
        key = 'res_'+k.split('/')[-1].split('.')[0].split('_')[-1]
        res_dict[key] = res_list
        print(f'{k}, {n}/{total_len}', end='\r')
        n += 1
    file = open(args.json_file, "w")
    json.dump(res_dict, file)
    file.close()


async def async_main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    tasks = asyncio.create_task(async_inference_detector(model, args.img))
    result = await asyncio.gather(tasks)
    # show the results
    show_result_pyplot(model, args.img, result[0], score_thr=args.score_thr)


if __name__ == '__main__':
    args = parse_args()
    if args.async_test:
        asyncio.run(async_main(args))
    else:
        main(args)
