import torch
from PIL import Image
import requests
from transformers import SamModel, SamProcessor

# device = "cuda" if torch.cuda.is_available() else "cpu"
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
generator = pipeline("mask-generation", model="facebook/sam-vit-huge", device='cuda:1')
from PIL import Image
import requests

img_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")

# plt.imshow(raw_image)

outputs = generator(raw_image, points_per_batch=64)
masks = outputs["masks"] #len 59, masks[0] (480,640)
# show_masks_on_image(raw_image, masks)

# plt.show()


