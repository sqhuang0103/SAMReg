import torch
from PIL import Image
import requests
from transformers import SamModel, SamProcessor
import cv2

# device = "cuda:1" if torch.cuda.is_available() else "cpu"
device = 'cuda:0'
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




from transformers import pipeline
generator = pipeline("mask-generation", model="facebook/sam-vit-huge", device=0)
from PIL import Image



class RoiMatching():
    def __init__(self,imgs1,imgs2,device):
        """

        :param imgs1: list of PIL images
        :param imgs2:
        """
        self.imgs1 = imgs1
        self.imgs2 = imgs2
        self.device = device
        self.masks1 = self.sam_everything(self.imgs1) # len(RM.masks1) 2; RM.masks1[0] dict; RM.masks1[0]['masks'] list
        self.masks2 = self.sam_everything(self.imgs2)

    def sam_everything(self,imgs):
        generator = pipeline("mask-generation", model="facebook/sam-vit-huge", device=self.device)
        outputs = generator(imgs, points_per_batch=64, pred_iou_thresh=0.90, stability_score_thresh=0.8)
        return outputs

    # def mask_criteria(self):
        


im1 = Image.open("/raid/shiqi/1B_B7_T.png").convert("RGB")
im2 = Image.open("/raid/shiqi/1B_B7_R.png").convert("RGB")
imgs = [im12]
RM = RoiMatching(imgs,imgs,device)
import pdb
pdb.set_trace()

