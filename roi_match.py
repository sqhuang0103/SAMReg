import torch
from PIL import Image
import requests
from transformers import SamModel, SamProcessor
import cv2
from scipy.spatial.distance import cosine

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
        outputs = generator(imgs, points_per_batch=64,pred_iou_thresh=0.90,stability_score_thresh=0.9,)
        return outputs
    def _mask_criteria(self, masks, v_min=200, v_max= 7000):
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

    def _roi_proto(self, image, masks):
        model = SamModel.from_pretrained("facebook/sam-vit-huge").to(self.device)
        processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
        inputs = processor(image, return_tensors="pt").to(self.device)
        # pixel_values" torch.size(1,3,1024,1024); "original_size" tensor([[834,834]]); 'reshaped_input_sizes' tensor([[1024, 1024]])
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

            # Compute mean for each channel, ignoring zeros
            tmp_emb[tmp_emb == 0] = np.nan  # Replace zeros with NaN for mean computation
            means = torch.nanmean(tmp_emb, dim=(2, 3))  # Compute means, ignoring NaN
            means[torch.isnan(means)] = 0  # Replace NaN with zeros
            embs.append(means)
        return embs

    def _cosine_similarity(self, vec1, vec2):
        # Ensure vec1 and vec2 are 2D tensors [1, N]
        vec1 = vec1.view(1, -1)
        vec2 = vec2.view(1, -1)
        # Using PyTorch's cosine_similarity. Need to unsqueeze to add batch dimension.
        return 1 - cosine_similarity(vec1, vec2).item()

    def _similarity_matrix(self, protos1, protos2):
        # Initialize similarity_matrix as a torch tensor
        similarity_matrix = torch.zeros(len(protos1), len(protos2), device=self.device)
        for i, vec_a in enumerate(protos1):
            for j, vec_b in enumerate(protos2):
                similarity_matrix[i, j] = self._cosine_similarity(vec_a, vec_b)
        # Normalize the similarity matrix
        sim_matrix = (similarity_matrix - similarity_matrix.min()) / (similarity_matrix.max() - similarity_matrix.min())
        return sim_matrix

    def _roi_match(self, matrix, masks1, masks2):
        index_pairs = []
        for _ in range(min(len(masks1), len(masks2))):
            max_sim_idx = torch.argmax(matrix).item()  # Get the index of the highest value as integer
            row = max_sim_idx // matrix.shape[1]  # Calculate row index
            col = max_sim_idx % matrix.shape[1]  # Calculate column index
            index_pairs.append((row, col))
            matrix[row, :] = -1  # Invalidate this row
            matrix[:, col] = -1  # Invalidate this column
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
        self.masks1 = self._sam_everything(self.img1)  # len(RM.masks1) 2; RM.masks1[0] dict; RM.masks1[0]['masks'] list
        self.masks2 = self._sam_everything(self.img2)
        self.masks1 = self._mask_criteria(self.masks1['masks'], v_min=self.v_min, v_max=self.v_max)
        self.masks2 = self._mask_criteria(self.masks2['masks'], v_min=self.v_min, v_max=self.v_max)

        if len(self.masks1) > 0 and len(self.masks2) > 0:
            self.embs1 = self._roi_proto(self.img1, self.masks1)  # device:cuda1
            self.embs2 = self._roi_proto(self.img2, self.masks2)
            self.sim_matrix = self._similarity_matrix(self.embs1, self.embs2)
            self.masks1, self.masks2 = self._roi_match(self.sim_matrix, self.masks1, self.masks2)


        # match self.mode:
        #     case 'embedding':
        #         if len(self.masks1) > 0 and len(self.masks2) > 0:
        #             self.embs1 = self._roi_proto(self.img1,self.masks1) #device:cuda1
        #             self.embs2 = self._roi_proto(self.img2,self.masks2)
        #             self.sim_matrix = self._similarity_matrix(self.embs1, self.embs2)
        #             self.masks1, self.masks2 = self._roi_match(self.sim_matrix,self.masks1,self.masks2)
        #     case 'overlaping':
        #         self._overlap_pair(self.masks1,self.masks2)


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



im1 = Image.open("/home/shiqi/SAMReg/example/pathology/1B_B7_R.png").convert("RGB")
im2 = Image.open("/home/shiqi/SAMReg/example/pathology/1B_B7_T.png").convert("RGB")
device='cuda:1'
RM = RoiMatching(im1,im2,device)
RM.get_paired_roi()
visualized_image_1 = visualize_masks(RM.masks1,im1)
cv2.imshow("Visualized Image_1", visualized_image_1)
visualized_image_2 = visualize_masks(RM.masks2,im2)
cv2.imshow("Visualized Image_2", visualized_image_2)

cv2.waitKey(0)
cv2.destroyAllWindows()

