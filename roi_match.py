import torch
from PIL import Image
import requests
from transformers import SamModel, SamProcessor
import cv2

device = "cuda" if torch.cuda.is_available() else "cpu"
# model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
# processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
#
# img_url = "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"
# raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
# input_points = [[[450, 600]]]  # 2D location of a window in the image
#
# inputs = processor(raw_image, input_points=input_points, return_tensors="pt").to(device)
# with torch.no_grad():
#     outputs = model(**inputs)
#
# masks = processor.image_processor.post_process_masks(
#     outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
# )
# scores = outputs.iou_scores

import numpy as np
import matplotlib.pyplot as plt
import gc



from transformers import pipeline
generator = pipeline("mask-generation", model="facebook/sam-vit-huge", device=0)
from PIL import Image
import requests

# img_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"
# raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
# new_image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/depth-estimation-example.jpg"
# outputs = generator(raw_image, points_per_batch=64)
# masks = outputs["masks"] #len 59, masks[0] (480,640), max True


class RoiMatching():
    def __init__(self,imgs1,imgs2,device):
        """

        :param imgs1: list of ndarray images
        :param imgs2:
        """
        self.imgs1 = [self.img_preprocess(item) for item in imgs1]
        self.imgs2 = [self.img_preprocess(item) for item in imgs2]
        self.device = device
        self.masks1 = self.sam_everything(self.imgs1)
        self.masks2 = self.sam_everything(self.imgs2)

    def img_preprocess(self,im):
        if len(im.shape) == 2:
            im = np.stack((im, im, im), axis=-1)
        # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        return im

    def sam_everything(self,imgs):
        generator = pipeline("mask-generation", model="facebook/sam-vit-huge", device=self.device)
        outputs = generator(imgs, points_per_batch=64)
        return outputs


im1 = Image.open("/raid/shiqi/1B_B7_T.png")
im2 = Image.open("/raid/shiqi/1B_B7_R.png")
imgs = [im1,im2]
RM = RoiMatching(imgs,imgs,device)
import pdb
pdb.set_trace()
