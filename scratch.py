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

    # Iterate through binary masks and overlay on the blank mask with random colors
    for idx, binary_mask in enumerate(binary_masks):
        # Generate a random color for each mask
        color = tuple(np.random.randint(0, 256, size=3).tolist())

        # Convert binary mask to uint8
        binary_mask = np.uint8(binary_mask)

        # Create a mask where binary mask is True
        fg_mask = np.where(binary_mask, 255, 0).astype(np.uint8)

        # Apply the foreground mask on the result with the random color
        mask[fg_mask > 0] = color

    # Add the mask on top of the background image
    result = cv2.addWeighted(background, 1, mask, 0.5, 0)

    return result

def _mask_criteria(masks, v_min=200, v_max= 7000):
    remove_list = set()
    for _i, mask in enumerate(masks):
        # print(mask['area'])
        if mask.sum() < v_min or mask.sum() > v_max:
            # if mask['segmentation'].sum() < 200 or mask['segmentation'].sum() > 20000:
            remove_list.add(_i)
    masks = [mask for idx, mask in enumerate(masks) if idx not in remove_list]
    n = len(masks)
    remove_list = set()
    for i in range(n):
        for j in range(i + 1, n):
            mask1, mask2 = masks[i], masks[j]
            intersection = (mask1 & mask2).sum()
            smaller_mask_area = min(masks[i].sum(), masks[j].sum())

            if smaller_mask_area > 0 and (intersection / smaller_mask_area) >= 0.9:
                if mask1.sum() < mask2.sum():
                    remove_list.add(i)
                else:
                    remove_list.add(j)
    return [mask for idx, mask in enumerate(masks) if idx not in remove_list]

def _maskselect(masks, v_min=200, v_max= 7000):
    remove_list = set()
    for _i, mask in enumerate(masks):
        # print(mask['area'])
        if mask['segmentation'].sum()< v_min or mask['segmentation'].sum() > v_max:
        # if mask['segmentation'].sum()< 200 or mask['segmentation'].sum() > 7000:
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
    return [mask['segmentation'] for idx, mask in enumerate(masks) if idx not in remove_list]


im1 = Image.open("/raid/candi/shiqi/slice_1_3.png").convert("RGB")
# im2 = Image.open("/raid/candi/shiqi/slice_1_1.png").convert("RGB")
# im1 = Image.open("/home/shiqi/SAMReg/example/cell/PNT1A_do_1_f00_01_01_R.png").convert("RGB")
im2 = Image.open("/home/shiqi/SAMReg/example/cell/PNT1A_do_1_f00_01_01_R.png").convert("RGB")
print(im1.size)
device='cuda:1'
from transformers import pipeline
generator = pipeline("mask-generation", model="facebook/sam-vit-huge", device=device)
outputs = generator(im1,points_per_batch=64,pred_iou_thresh=0.90,stability_score_thresh=0.9,)
import pdb
pdb.set_trace()
masks = outputs["masks"]
masks = _mask_criteria(masks)

from model.segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
# im1_cv = cv2.imread("/raid/candi/shiqi/slice_1_3.png")
im1_cv = cv2.imread("/home/shiqi/SAMReg/example/cell/PNT1A_do_1_f00_01_01_R.png",0)
im2_cv = cv2.imread("/home/shiqi/SAMReg/example/cell/PNT1A_do_1_f00_01_01_T.png",0)
# im2_cv = cv2.imread("/raid/candi/shiqi/slice_1_1.png")
sam = sam_model_registry['vit_h'](checkpoint='/raid/candi/shiqi/sam_pretrained/sam_vit_h_4b8939.pth')
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(model=sam,
                                                   pred_iou_thresh=0.90, #0.90
                                                   # min_mask_region_area=150
                                                   # points_per_side=32
                                                   stability_score_thresh=0.9, #0.8
                                                   )
masks_sam = mask_generator.generate(im1_cv)
masks_sam = _maskselect(masks_sam)

visualized_image = visualize_masks(masks, im1)
cv2.imshow("Visualized Image", visualized_image)

visualized_image_sam = visualize_masks(masks_sam, im1)
cv2.imshow("Visualized Image_SAM", visualized_image_sam)


cv2.waitKey(0)
cv2.destroyAllWindows()