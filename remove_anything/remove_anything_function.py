import torch
import sys
import argparse
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
import os

from remove_anything.sam_segment import predict_masks_with_sam
from remove_anything.lama_inpaint import inpaint_img_with_lama
from remove_anything.utils import dilate_mask

def test_mask(img, point_coords, point_labels,
                sam_model_type, sam_ckpt):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    latest_coords = point_coords
    masks, _, _ = predict_masks_with_sam(
        img,
        [latest_coords],
        point_labels,
        model_type=sam_model_type,
        ckpt_p=sam_ckpt,
        device=device,
    )
    masks = masks.astype(np.uint8) * 255
    
    return masks

def remove_anything(img, dilate_kernel_size,
                    lama_config, lama_ckpt, masks):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if dilate_kernel_size is not None:
        masks = [dilate_mask(mask, dilate_kernel_size) for mask in masks]

    res_inpaint_list = []
    for idx, mask in enumerate(masks):
        img_inpainted = inpaint_img_with_lama(
            img, mask, lama_config, lama_ckpt, device=device)
        res_inpaint_list.append(img_inpainted.astype(np.uint8))
    
    
    mask_centers = []
    for idx, mask in enumerate(masks):
        white_points = np.argwhere(mask == 255)
        x_min = white_points[:, 1].min()
        x_max = white_points[:, 1].max()
        y_min = white_points[:, 0].min()
        y_max = white_points[:, 0].max()
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2
        mask_centers.append((center_x, center_y))
    
    return mask_centers, res_inpaint_list

