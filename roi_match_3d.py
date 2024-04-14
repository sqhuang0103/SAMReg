from transformers import pipeline
from PIL import Image
import numpy as np
import cv2
import torch
from torch.nn.functional import cosine_similarity
from utils import Metric, Vis, Vis_cv2
import time
import matplotlib.pyplot as plt
import torch.nn.functional as F
import random
device = 'cpu'

class ROIMatching():
    def __init__(self):
        def __init__(self, img1, img2, device='cuda:1', v_min=200, v_max=7000, mode='embedding',
                     url="facebook/sam-vit-huge", sim_criteria=0.8):
            """
            Initialize
            :param img1: PIL image
            :param img2:
            """
            self.img1 = img1
            self.img2 = img2
            self.device = device
            self.v_min = v_min
            self.v_max = v_max
            self.mode = mode
            self.url = url
            self.sim_criteria = sim_criteria

        def _get_model(self):
            pass
        def _get_prompt_mask(self):
            pass
