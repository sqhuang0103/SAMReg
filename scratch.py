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



# from transformers import pipeline
# generator = pipeline("mask-generation", model="facebook/sam-vit-huge", device='cuda:1')
# from PIL import Image
# import requests
#
# img_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"
# raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
#
# # plt.imshow(raw_image)
#
# outputs = generator(raw_image, points_per_batch=64)
# masks = outputs["masks"] #len 59, masks[0] (480,640)
# # show_masks_on_image(raw_image, masks)
#
# # plt.show()
import cv2
import PIL
import numpy as np
def visualize_masks(binary_masks, pil_image):
    # Convert PIL image to numpy array
    background = np.array(pil_image)

    # Convert RGB to BGR (OpenCV uses BGR color format)
    background = cv2.cvtColor(background, cv2.COLOR_RGB2BGR)

    # Create a blank mask
    mask = np.zeros_like(background)

    # Define colors for different masks
    colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]  # You can add more colors as needed

    # Iterate through binary masks and overlay on the blank mask with different colors
    for idx, binary_mask in enumerate(binary_masks):
        # Convert binary mask to uint8
        binary_mask = np.uint8(binary_mask)

        # Create a mask where binary mask is True
        fg_mask = np.where(binary_mask, 255, 0).astype(np.uint8)

        # Apply the foreground mask on the result with the corresponding color
        mask[fg_mask > 0] = colors[idx % len(colors)]

    # Add the mask on top of the background image
    result = cv2.addWeighted(background, 1, mask, 0.5, 0)

    return result

im1 = Image.open("/raid/shiqi/slice_1_3.png").convert("RGB")
im2 = Image.open("/raid/shiqi/slice_1_1.png").convert("RGB")
device='cuda:0'
from transformers import pipeline
generator = pipeline("mask-generation", model="facebook/sam-vit-huge", device='cuda:0')
outputs = generator(im1, points_per_batch=64)
masks = outputs["masks"]
visualized_image = visualize_masks(masks, im1)
cv2.imshow("Visualized Image", visualized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()