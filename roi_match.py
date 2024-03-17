import torch
from PIL import Image
import requests
from transformers import SamModel, SamProcessor
import cv2

# device = "cuda:1" if torch.cuda.is_available() else "cpu"
device = 'cpu'
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
print('start')


class RoiMatching():
    def __init__(self,img1,img2,device='cuda:1', v_min=200, v_max= 7000, mode = 'embedding'):
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

    def _sam_everything(self,imgs):
        generator = pipeline("mask-generation", model="facebook/sam-vit-huge", device=self.device)
        outputs = generator(imgs, points_per_batch=64, pred_iou_thresh=0.90, stability_score_thresh=0.8)
        return outputs
    def _mask_criteria(self, masks, v_min=200, v_max= 7000):
        remove_list = set()
        for _i, mask in enumerate(masks):
            # print(mask['area'])
            if mask['segmentation'].sum() < v_min or mask['segmentation'].sum() > v_max:
                # if mask['segmentation'].sum() < 200 or mask['segmentation'].sum() > 20000:
                remove_list.add(_i)
        masks = [mask for idx, mask in enumerate(masks) if idx not in remove_list]
        n = len(masks)
        remove_list = set()
        for i in range(n):
            for j in range(i + 1, n):
                mask1, mask2 = masks[i]['segmentation'], masks[j]['segmentation']
                intersection = (mask1 & mask2).sum()
                smaller_mask_area = min(masks[i]['area'], masks[j]['area'])

                if smaller_mask_area > 0 and (intersection / smaller_mask_area) >= 0.9:
                    if mask1.sum() < mask2.sum():
                        remove_list.add(i)
                    else:
                        remove_list.add(j)
        return [mask for idx, mask in enumerate(masks) if idx not in remove_list]

    def _roi_proto(self, image, masks):
        model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
        processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
        inputs = processor(image, return_tensors="pt").to(device)
        image_embeddings = model.get_image_embeddings(inputs["pixel_values"])

    def get_paired_roi(self):
        self.masks1 = self._sam_everything(self.img1)  # len(RM.masks1) 2; RM.masks1[0] dict; RM.masks1[0]['masks'] list
        self.masks2 = self._sam_everything(self.img2)
        self.masks1 = self._mask_criteria(self.masks1, v_min=self.v_min, v_max=self.v_max)
        self.masks2 = self._mask_criteria(self.masks2, v_min=self.v_min, v_max=self.v_max)

        match self.mode:
            case 'embedding':
                if len(self.masks1) > 1 and len(self.masks2) > 1:
                    self.sam_pair_embedding(self.masks1[:-1],self.masks2[:-1])
                    self.embedding_pair(self.m1_embs,self.m2_embs)



im1 = Image.open("/raid/shiqi/1B_B7_T.png").convert("RGB")
im2 = Image.open("/raid/shiqi/1B_B7_R.png").convert("RGB")

print('here')
def roi_proto(image, device = 'cpu',masks=None):
    print('here')
    model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
    processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
    inputs = processor(image, return_tensors="pt").to(device)
    image_embeddings = model.get_image_embeddings(inputs["pixel_values"])
    import pdb
    pdb.set_trace()
roi_proto(im1)
# RM = RoiMatching(imgs,imgs,device)

