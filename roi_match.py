import torch
from PIL import Image
import requests
from transformers import SamModel, SamProcessor
import cv2
from scipy.spatial.distance import cosine
import random

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
from PIL import Image
import numpy as np
import cv2
import torch
from torch.nn.functional import cosine_similarity
from utils import Metric, Vis, Vis_cv2
import time
import matplotlib.pyplot as plt


class RoiMatching():
    def __init__(self,img1,img2,device='cuda:1', v_min=200, v_max= 7000, mode = 'embedding', url = "facebook/sam-vit-huge", sim_criteria = 0.8):
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

    def _sam_everything(self,imgs):
        generator = pipeline("mask-generation", model=self.url, device=self.device)
        outputs = generator(imgs, points_per_batch=64,pred_iou_thresh=0.90,stability_score_thresh=0.9,)
        # outputs = generator(imgs, points_per_batch=64,stability_score_thresh=0.7,) #medsam
        return outputs
    def _mask_criteria(self, masks, v_min=200, v_max= 7000):
        remove_list = set()
        for _i, mask in enumerate(masks):
            if mask.sum() < v_min or mask.sum() > v_max:
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

    def _roi_proto(self, image, masks):
        # im = np.array(image)
        # from model.segment_anything import sam_model_registry, SamPredictor
        # sam = sam_model_registry["vit_h"](checkpoint='/raid/candi/shiqi/sam_pretrained/sam_vit_h_4b8939.pth')
        # sam.to(device=self.device)
        # predictor = SamPredictor(sam)
        # predictor.set_image(im)
        # image_embeddings = predictor.get_image_embedding() # .cpu().numpy()  # (1, 256, 64, 64)

        model = SamModel.from_pretrained(self.url).to(self.device) #"facebook/sam-vit-huge" "wanglab/medsam-vit-base"
        processor = SamProcessor.from_pretrained(self.url)
        inputs = processor(image, return_tensors="pt").to(self.device)
        # # pixel_values" torch.size(1,3,1024,1024); "original_size" tensor([[834,834]]); 'reshaped_input_sizes' tensor([[1024, 1024]])
        image_embeddings = model.get_image_embeddings(inputs["pixel_values"])
        # torch.Size([1, 256, 64, 64])
        embs = []
        for _m in masks:
            # Convert mask to uint8, resize, and then back to boolean
            tmp_m = _m.astype(np.uint8)
            tmp_m = cv2.resize(tmp_m, (64, 64), interpolation=cv2.INTER_NEAREST)
            tmp_m = torch.tensor(tmp_m.astype(bool), device=self.device,
                                 dtype=torch.float32)  # Convert to tensor and send to CUDA
            tmp_m = tmp_m.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions to match emb1

            # Element-wise multiplication with emb1
            tmp_emb = image_embeddings * tmp_m
            # (1,256,64,64)

            tmp_emb[tmp_emb == 0] = torch.nan
            emb = torch.nanmean(tmp_emb, dim=(2, 3))
            emb[torch.isnan(emb)] = 0
            embs.append(emb)
        return embs

    def _cosine_similarity(self, vec1, vec2):
        # Ensure vec1 and vec2 are 2D tensors [1, N]
        vec1 = vec1.view(1, -1)
        vec2 = vec2.view(1, -1)
        return cosine_similarity(vec1, vec2).item()

    def _similarity_matrix(self, protos1, protos2):
        # Initialize similarity_matrix as a torch tensor
        similarity_matrix = torch.zeros(len(protos1), len(protos2), device=self.device)
        for i, vec_a in enumerate(protos1):
            for j, vec_b in enumerate(protos2):
                similarity_matrix[i, j] = self._cosine_similarity(vec_a, vec_b)
                # print('RM: ', vec_a.max(), vec_b.max(),similarity_matrix[i,j])
        # Normalize the similarity matrix
        sim_matrix = (similarity_matrix - similarity_matrix.min()) / (similarity_matrix.max() - similarity_matrix.min())
        return similarity_matrix

    def _roi_match(self, matrix, masks1, masks2, sim_criteria=0.8):
        index_pairs = []
        while torch.any(matrix > sim_criteria):
            max_idx = torch.argmax(matrix)
            max_sim_idx = (max_idx // matrix.shape[1], max_idx % matrix.shape[1])
            if matrix[max_sim_idx[0], max_sim_idx[1]] > sim_criteria:
                index_pairs.append(max_sim_idx)
            matrix[max_sim_idx[0], :] = -1
            matrix[:, max_sim_idx[1]] = -1
        masks1_new = []
        masks2_new = []
        for i, j in index_pairs:
            masks1_new.append(masks1[i])
            masks2_new.append(masks2[j])
        return masks1_new, masks2_new

    def _overlap_pair(self, masks1,masks2):
        self.masks1_cor = []
        self.masks2_cor = []
        k = 0
        for mask in masks1[:-1]:
            k += 1
            print('mask1 {} is finding corresponding region mask...'.format(k))
            m1 = mask
            a1 = mask.sum()
            v1 = np.mean(np.expand_dims(m1, axis=-1) * self.im1)
            overlap = m1 * masks2[-1].astype(np.int64)
            # print(np.unique(overlap))
            if (overlap > 0).sum() / a1 > 0.3:
                counts = np.bincount(overlap.flatten())
                # print(counts)
                sorted_indices = np.argsort(counts)[::-1]
                top_two = sorted_indices[1:3]
                # print(top_two)
                if top_two[-1] == 0:
                    cor_ind = 0
                elif abs(counts[top_two[-1]] - counts[top_two[0]]) / max(counts[top_two[-1]], counts[top_two[0]]) < 0.2:
                    cor_ind = 0
                else:
                    # cor_ind = 0
                    m21 = masks2[top_two[0]-1]
                    m22 = masks2[top_two[1]-1]
                    a21 = masks2[top_two[0]-1].sum()
                    a22 = masks2[top_two[1]-1].sum()
                    v21 = np.mean(np.expand_dims(m21, axis=-1)*self.im2)
                    v22 = np.mean(np.expand_dims(m22, axis=-1)*self.im2)
                    if np.abs(a21-a1) > np.abs(a22-a1):
                        cor_ind = 0
                    else:
                        cor_ind = 1
                    print('area judge to cor_ind {}'.format(cor_ind))
                    if np.abs(v21-v1) < np.abs(v22-v1):
                        cor_ind = 0
                    else:
                        cor_ind = 1
                    # print('value judge to cor_ind {}'.format(cor_ind))
                # print('mask1 {} has found the corresponding region mask: mask2 {}'.format(k, top_two[cor_ind]))

                self.masks2_cor.append(masks2[top_two[cor_ind] - 1])
                self.masks1_cor.append(mask)
        # return masks1_new, masks2_new

    def get_paired_roi(self):
        batched_imgs = [self.img1, self.img2]
        batched_outputs = self._sam_everything(batched_imgs)
        self.masks1, self.masks2 = batched_outputs[0], batched_outputs[1] # 16.554s

        # self.masks1 = self._sam_everything(self.img1)  # len(RM.masks1) 2; RM.masks1[0] dict; RM.masks1[0]['masks'] list
        # self.masks2 = self._sam_everything(self.img2)

        self.masks1 = self._mask_criteria(self.masks1['masks'], v_min=self.v_min, v_max=self.v_max)
        self.masks2 = self._mask_criteria(self.masks2['masks'], v_min=self.v_min, v_max=self.v_max)

        match self.mode:
            case 'embedding':
                if len(self.masks1) > 0 and len(self.masks2) > 0:
                    self.embs1 = self._roi_proto(self.img1,self.masks1) #device:cuda1
                    self.embs2 = self._roi_proto(self.img2,self.masks2) # 6.752s
                    self.sim_matrix = self._similarity_matrix(self.embs1, self.embs2)
                    self.masks1, self.masks2 = self._roi_match(self.sim_matrix,self.masks1,self.masks2,self.sim_criteria)
            case 'overlaping':
                self._overlap_pair(self.masks1,self.masks2)

    def get_prompt_roi(self):
        self.model = SamModel.from_pretrained(self.url).to(self.device)  # "facebook/sam-vit-huge" "wanglab/medsam-vit-base"
        self.processor = SamProcessor.from_pretrained(self.url)
        batched_imgs = [self.img1, self.img2]
        batched_outputs = self._get_image_embedding(batched_imgs)

        H,W = self.img1.size
        # point = self._get_random_coordinates((H,W),1) # array([[464, 360]])
        point = np.array([[383,543]])
        self.emb1, self.emb2 = batched_outputs[0].unsqueeze(0), batched_outputs[1].unsqueeze(0) # torch.Size([256, 64, 64])
        masks_f, scores_f = self._get_prompt_mask(self.img1, self.emb1, input_points=[point], labels=[1])
        # m[0].shape: torch.Size([1, 3, 834, 834]); tensor([[[0.9626, 0.9601, 0.7076]]], device='cuda:0')
        mask_f = masks_f[0][:,torch.argmax(scores_f[0][0]),:,:]
        return m,s, point



    def _get_random_coordinates(self, shape, n_points, mask=None):
        """
        Generate random coordinates within a given shape. If a mask is provided,
        the points are generated within the non-zero regions of the mask.

        Parameters:
        - shape: tuple of (H, W), the dimensions of the space where points are generated.
        - n_points: int, the number of points to generate.
        - mask: ndarray or None, the mask within which to generate points, or None to ignore.

        Returns:
        - coordinates: ndarray of shape (n_points, 2), each row is a coordinate (y, x).
        """
        H, W = shape
        if mask is None:
            # Generate random coordinates over the entire shape
            coordinates = np.column_stack((np.random.randint(0, H, n_points),
                                           np.random.randint(0, W, n_points)))
        else:
            # Find indices where mask is greater than 0
            y_indices, x_indices = np.where(mask > 0)
            if len(y_indices) < n_points:
                raise ValueError("Not enough points within the mask to generate the requested number of coordinates.")

            # Randomly choose n_points indices from the non-zero regions of the mask
            chosen_indices = np.random.choice(len(y_indices), n_points, replace=False)
            coordinates = np.column_stack((y_indices[chosen_indices], x_indices[chosen_indices]))

        return coordinates

    def _get_image_embedding(self, image):
        inputs = self.processor(image, return_tensors="pt").to(self.device)
        # # pixel_values" torch.size(1,3,1024,1024); "original_size" tensor([[834,834]]); 'reshaped_input_sizes' tensor([[1024, 1024]])
        image_embeddings = self.model.get_image_embeddings(inputs["pixel_values"])
        return image_embeddings

    def _get_prompt_mask(self, image, image_embeddings, input_points, labels, input_boxes=None):
        if input_boxes is not None:
            inputs = self.processor(image, input_boxes=[input_boxes], input_points=[input_points], input_labels=[labels],
                               return_tensors="pt").to(device)
        else:
            inputs = self.processor(image, input_points=[input_points], input_labels=[labels],
                                    return_tensors="pt").to(device)

        inputs.pop("pixel_values", None)
        inputs.update({"image_embeddings": image_embeddings})
        with torch.no_grad():
            outputs = self.model(**inputs)

        masks = self.processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(),
                                                             inputs["reshaped_input_sizes"].cpu())
        scores = outputs.iou_scores
        return masks, scores


def visualize_masks(image1, masks1, image2, masks2):
    # Convert PIL images to numpy arrays
    background1 = np.array(image1)
    background2 = np.array(image2)

    # Convert RGB to BGR (OpenCV uses BGR color format)
    background1 = cv2.cvtColor(background1, cv2.COLOR_RGB2BGR)
    background2 = cv2.cvtColor(background2, cv2.COLOR_RGB2BGR)

    # Create a blank mask for each image
    mask1 = np.zeros_like(background1)
    mask2 = np.zeros_like(background2)

    distinct_colors = [
        (255, 0, 0),  # Red
        (0, 255, 0),  # Green
        (0, 0, 255),  # Blue
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 0, 0),  # Maroon
        (0, 128, 0),  # Olive
        (0, 0, 128),  # Navy
        (128, 128, 0),  # Teal
        (128, 0, 128),  # Purple
        (0, 128, 128),  # Gray
        (192, 192, 192)  # Silver
    ]

    def random_color():
        """Generate a random color with high saturation and value in HSV color space."""
        hue = random.randint(0, 179)  # Random hue value between 0 and 179 (HSV uses 0-179 range)
        saturation = random.randint(200, 255)  # High saturation value between 200 and 255
        value = random.randint(200, 255)  # High value (brightness) between 200 and 255
        color = np.array([[[hue, saturation, value]]], dtype=np.uint8)
        return cv2.cvtColor(color, cv2.COLOR_HSV2BGR)[0][0]


    # Iterate through mask lists and overlay on the blank masks with different colors
    for idx, (mask1_item, mask2_item) in enumerate(zip(masks1, masks2)):
        # color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        # color = distinct_colors[idx % len(distinct_colors)]
        color = random_color()
        # Convert binary masks to uint8
        mask1_item = np.uint8(mask1_item)
        mask2_item = np.uint8(mask2_item)

        # Create a mask where binary mask is True
        fg_mask1 = np.where(mask1_item, 255, 0).astype(np.uint8)
        fg_mask2 = np.where(mask2_item, 255, 0).astype(np.uint8)

        # Apply the foreground masks on the corresponding masks with the same color
        mask1[fg_mask1 > 0] = color
        mask2[fg_mask2 > 0] = color

    # Add the masks on top of the background images
    result1 = cv2.addWeighted(background1, 1, mask1, 0.5, 0)
    result2 = cv2.addWeighted(background2, 1, mask2, 0.5, 0)

    return result1, result2


def visualize_masks_with_scores(image, masks, scores, points, labels):
    """
    Visualize masks with their scores on the original image.

    Parameters:
    - image: PIL image with size (H, W)
    - masks: torch tensor of shape [1, 3, H, W]
    - scores: torch tensor of scores with shape [1, 3]
    """
    # Convert PIL image to NumPy array
    image_np = np.array(image)

    # Move masks and scores to CPU and convert to NumPy
    masks_np = masks.cpu().numpy().squeeze(0)  # Shape [3, H, W]
    scores_np = scores.cpu().numpy().squeeze(0)  # Shape [3]

    # Set up the plot
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    for i in range(3):
        ax = axs[i]
        score = scores_np[i]
        mask = masks_np[i]
        # Create an RGBA image for the mask
        mask_image = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
        mask_image[..., 3] = mask * 255  # Alpha channel
        # Overlay the mask on the image
        ax.imshow(image_np)
        ax.imshow(mask_image, cmap='jet', alpha=0.5)
        ax.scatter(points[:, 0], points[:, 1], c='red', marker='o', label='Scatter Points')
        ax.set_title(f'Score: {score:.4f}')
        ax.axis('off')
    plt.tight_layout()
    plt.show()



im1 = Image.open("/home/shiqi/SAMReg/example/pathology/1B_B7_R.png").convert("RGB")
im2 = Image.open("/home/shiqi/SAMReg/example/pathology/1B_B7_T.png").convert("RGB")
device='cuda:0'
url="facebook/sam-vit-huge" #"facebook/sam-vit-huge" "wanglab/medsam-vit-base"
start_time = time.time()
RM = RoiMatching(im1,im2,device,url=url)

# transformers SAM implementation
# RM.get_paired_roi()
m,s, p = RM.get_prompt_roi()
print(p)
end_time = time.time()
inference_time = end_time - start_time
print(f"Inference Time: {inference_time:.3f} seconds")
visualize_masks_with_scores(im1,m[0],s[0], p, labels=[1])
# visualized_image1, visualized_image2 = visualize_masks(im1, RM.masks1, im2, RM.masks2)

# SAM repo implementation
# im1 = cv2.imread("/home/shiqi/SAMReg/example/prostate_2d/image1.png")
# im2 = cv2.imread("/home/shiqi/SAMReg/example/prostate_2d/image2.png")
# from model.segment_anything import sam_model_registry, SamPredictor
# from demo import get_pair_masks
# from model.pair2d import PairMasks
# sam = sam_model_registry["vit_h"](checkpoint='/raid/candi/shiqi/sam_pretrained/sam_vit_h_4b8939.pth')
# sam.to(device=device)
# PairM = PairMasks(sam, im1, im2, mode='embedding')
# masks1 = PairM.masks1_cor[:-1]
# masks2 = PairM.masks2_cor[:-1]
# masks1 = [mask['segmentation'] for mask in masks1]
# masks2 = [mask['segmentation'] for mask in masks2]
# visualized_image1, visualized_image2 = visualize_masks(im1, masks1, im2, masks2)


# cv2.imshow("Visualized Image 1", visualized_image1)
# cv2.imshow("Visualized Image 2", visualized_image2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

