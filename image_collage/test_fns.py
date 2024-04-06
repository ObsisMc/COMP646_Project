import cv2
import numpy as np


class TestImageProcesser:
    def __init__(self) -> None:
        self.pos = None
        self.mask = None
        self.img_masked = None
    
    def click_process(self, img: np.ndarray, coords: list):
        """
        img: RGB, (H, W, C)
        coords: (H, W)
        """
        
        if self.mask is not None:
            img[self.pos[0]: self.pos[1], self.pos[2]: self.pos[3]] = self.img_masked
            self.mask = self.pos = self.img_masked = None
        
        radius = 10
        color = (255, 0, 0)
        line_type = -1
        x, y = coords
        
        self.pos = (x - radius, x + radius, y - radius, y + radius)
        self.mask = np.zeros(img.shape[:2]).astype(bool)
        self.mask[self.pos[0]: self.pos[1], self.pos[2]: self.pos[3]] = True
        self.img_masked = img[self.pos[0]: self.pos[1], self.pos[2]: self.pos[3]].copy()
        
        cv2.circle(img, (y, x), radius, color, line_type)
        
        return img
    
    def replace(self, img, text):

        img[self.mask] = np.array([255,255,255])
        
        self.mask = None
        self.img_masked = None
        return img