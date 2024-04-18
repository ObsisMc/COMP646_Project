import cv2
import numpy as np
import matplotlib.pyplot as plt
from remove_anything.remove_anything_function import remove_anything, test_mask

import time



class TestImageProcesser:
    def __init__(self) -> None:
        self.mask = None
        self.original = None
        
        self.sam_model_type = "vit_h"
        self.sam_ckpt = "./remove_anything/pretrained_models/sam_vit_h_4b8939.pth"
        self.lama_config = "./remove_anything/lama/configs/prediction/default.yaml"
        self.lama_ckpt = "./remove_anything/pretrained_models/big-lama"
        self.dilate_kernel_size = 15
        
    
    def click_process(self, img: np.ndarray, coords: list):
        """
        img: RGB, (H, W, C)
        coords: (H, W)
        """
        print(img.shape)
        # cv2.imwrite("test.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
        if self.mask is not None:
            img = self.original
            self.mask = self.original = None
        
        
        masks = test_mask(img, coords, [1], self.sam_model_type, self.sam_ckpt)
        mask = masks[1]
        self.mask = mask == 255
        
        # store original img
        self.original = img.copy()
        
        # paint mask
        colored_mask = np.zeros_like(img)
        colored_mask[self.mask] = (255, 0, 0) # red
        img = cv2.addWeighted(img, 1, colored_mask, 0.65, 0)
        
        return img
    
    def replace(self, img, text):

        img[self.mask] = np.array([255,255,255])
        
        # remove_anything(img, dilate_kernel_size, lama_config, lama_ckpt, masks)
        
        self.mask = None
        self.img_masked = None
        return img