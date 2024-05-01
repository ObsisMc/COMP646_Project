import cv2
import numpy as np
import matplotlib.pyplot as plt
from remove_anything.remove_anything_function import remove_anything, test_mask
from image_collage.clip import CLIPModelWrapper

import torch
import torch.nn.functional as F



class ImageProcesser:
    def __init__(self, segments_dir, embed_path) -> None:
        self.mask = None
        self.original = None
        
        self.sam_model_type = "vit_h"
        self.sam_ckpt = "./remove_anything/pretrained_models/sam_vit_h_4b8939.pth"
        self.lama_config = "./remove_anything/lama/configs/prediction/default.yaml"
        self.lama_ckpt = "./remove_anything/pretrained_models/big-lama"
        self.dilate_kernel_size = 15
        
        self.clip_wrapper = CLIPModelWrapper(image_dir=segments_dir, csv_file=embed_path)
        
    
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
        
        
        masks = test_mask(img, coords[::-1], [1], self.sam_model_type, self.sam_ckpt)
        self.mask = masks[1]
        
        # store original img
        self.original = img.copy()
        
        # paint mask
        colored_mask = np.zeros_like(img)
        colored_mask[self.mask == 255] = (255, 0, 0) # red
        img = cv2.addWeighted(img, 1, colored_mask, 0.65, 0)
        
        return img
    
    def replace(self, img, text_prompt, mode=2):
        print(f"text prompt:", text_prompt)
        if self.original is None or text_prompt in ["", None]:
            return img
        
        H, W, C = img.shape
        
        # get segments
        segments = self.clip_wrapper.find_top_similar_images(text_prompt, return_matrix=True)
        segment = segments[0]
        
        # remove original object
        mask_centers, res_inpaint_list, mask_bboxs = remove_anything(self.original, self.dilate_kernel_size, self.lama_config, self.lama_ckpt, [self.mask])
        (y_center, x_center), img, bbox = mask_centers[0], res_inpaint_list[0], mask_bboxs[0]
        print(x_center, y_center)
        
        # resize
        h_o, w_o = bbox[3] - bbox[1], bbox[2] - bbox[0]
        h_s, w_s = segment.shape[:2]
        print(h_o, w_o)
        print(h_s, w_s)
        ratio = np.sqrt(h_o * w_o / h_s / w_s)
        h_r, w_r = int(h_s * ratio), int(w_s * ratio)
        
        x_min, y_min = x_center - h_r // 2, y_center - w_r // 2
        x_max, y_max = x_center + h_r // 2, y_center + w_r // 2
        if x_min < 0: x_min = 0
        if y_min < 0: y_min = 0
        if x_max >= H: x_max = H - 1
        if y_max >= W: y_max = W - 1
        h_r, w_r = x_max - x_min, y_max - y_min
        
        print(f"resize from {segment.shape[:2]} to {h_r, w_r}")
        segment_resize = cv2.resize(segment, (w_r, h_r))
        
        # replace
        if mode == 1:
            mask = np.zeros_like(segment_resize)
            mask[segment_resize.sum(axis=-1) != 0] = 1
            mask *= 255
            
            img = cv2.seamlessClone(segment_resize, img, mask, ((y_max + y_min) // 2, (x_max + x_min) // 2), flags=cv2.NORMAL_CLONE)
            
        elif mode == 2:
            mask = np.zeros(segment_resize.shape[:2])
            mask[segment_resize.sum(axis=-1) != 0] = 1
            mask = mask.astype(bool)
            
            bg_crop = img[x_min:x_max, y_min:y_max]
            bg_crop[mask] = segment_resize[mask]
            print(f"bg crop size: {bg_crop.shape}")
            bg_crop_smoothed = self.smoothize(bg_crop)
            print(f"bg crop smooth size: {bg_crop_smoothed.shape}")
            
            bg_h, bg_W, _ = bg_crop_smoothed.shape
            img[x_min + 1: x_max - 1, y_min + 1: y_max - 1] = bg_crop_smoothed[1:bg_h - 1, 1:bg_W - 1]
        
        else:
            mask = np.zeros(segment_resize.shape[:2])
            mask[segment_resize.sum(axis=-1) != 0] = 1
            bg_mask = np.pad(mask, ((x_min, H - x_max), (y_min, W - y_max))).astype(bool)
            mask = mask.astype(bool)
            img[bg_mask] = segment_resize[mask]
        
        
        # test_img = np.zeros_like(img)
        # test_img[replace_mask_whole] = (255, 255, 255)
        # cv2.imwrite("test.png", test_img)
        
        self.mask = None
        self.original = None
        
        return img
    
    
    def smoothize(self, img: np.ndarray):
        kernel_size = 3
        stride = 1
        
        # Convert the image to a tensor
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
        img_tensor = img_tensor.to(torch.float32)  # Ensure the tensor is in float32 for the AvgPool operation
        img_tensor = img_tensor.unsqueeze(0)

        # Check if image tensor needs to be normalized (having pixel values between 0 and 1)
        if img_tensor.max() > 1.0:
            img_tensor /= 255.0

        # Apply AvgPool2d with kernel size 3, stride 1, and padding 1 to maintain the original image size
        smoothed_img_tensor = F.avg_pool2d(img_tensor, kernel_size, stride, padding=1)

        # Convert the smoothed tensor back to a PIL image with proper scaling
        smoothed_img = (smoothed_img_tensor.squeeze(0).permute(1, 2, 0).clamp(0, 1) * 255).byte().numpy()
        
        return smoothed_img
    
    def reset(self):
        self.mask = self.original = None