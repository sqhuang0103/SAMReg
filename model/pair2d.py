import os
# from dataset.datasets import load_data_volume
import argparse
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
from scipy.spatial.distance import cosine
import matplotlib.colors as mcolors
from transformers import SamModel, SamProcessor, pipeline
import cv2
import random
import numpy as np
import torch
from torch.nn.functional import cosine_similarity


class PairMasks():
    def __init__(self, im1, im2, args, mode = 'embedding'):
        sam = sam_model_registry[args.sam_mode](checkpoint=args.sam_checkpoint)
        sam.to(device=args.device)
        self.sam = sam
        self.im1 = self.img_preprocess(im1)
        self.im2 = self.img_preprocess(im2)
        self.masks1 = self.sam_everything(self.im1, v_min=200, v_max=7000) #last one is a multi-object mask #200
        self.masks2 = self.sam_everything(self.im2, v_min=200, v_max=7000)
        if mode == 'embedding':
            if len(self.masks1) > 1 and len(self.masks2) > 1:
                self.sam_pair_embedding(self.masks1[:-1],self.masks2[:-1])
                self.embedding_pair(self.m1_embs,self.m2_embs)
            else:
                self.masks1_cor = [{'segmentation': np.zeros_like(self.im1[:,:,0])}]
                self.masks2_cor = [{'segmentation': np.zeros_like(self.im2[:,:,0])}]
        elif mode == 'overlaping':
            self.overlap_pair(self.masks1,self.masks2)
        else:
            self.masks1_cor = None
            self.masks2_cor = None



    def img_preprocess(self,im):
        if len(im.shape) == 2:
            im = np.stack((im, im, im), axis=-1)
        # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        return im

    def _maskfuse(self, masks):
        img = np.zeros(masks[0]['segmentation'].shape)
        for i in range(len(masks)):
            _m = np.uint8(masks[i]['segmentation'])
            if (_m * img).sum() > 0:
                _m = _m - np.uint8(_m * (img > 0))
            img += _m * (i + 1)
            if img.max() > len(masks):
                import pdb
                pdb.set_trace()
        return img

    def _maskselect(self, masks, v_min=200, v_max= 7000):
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
        return [mask for idx, mask in enumerate(masks) if idx not in remove_list]

    def sam_prompt(self,img,points,labels):
        predictor = SamPredictor(self.sam)
        predictor.set_image(img)
        point_coords = np.array(points) # np.array([[12,94],[23，26]])
        point_labels = np.array(labels) # np.array([1，0])

        embedding = predictor.get_image_embedding().cpu().numpy()
        masks, scores, logits = predictor.predict(point_coords=point_coords, point_labels=point_labels)

        return masks, scores, logits, embedding


    def sam_everything(self,img, v_min=200, v_max=7000):
        mask_generator = SamAutomaticMaskGenerator(model=self.sam,
                                                   pred_iou_thresh=0.90, #0.90
                                                   # min_mask_region_area=150
                                                   # points_per_side=32
                                                   stability_score_thresh=0.9, #0.8
                                                   )
        masks = mask_generator.generate(img)
        masks = self._maskselect(masks, v_min, v_max)
        if len(masks) > 0:
            masks.append({'segmentation': self._maskfuse(masks)})
        else:
            print('segmentation fail in this slice!!')
            masks.append({'segmentation': np.zeros_like(img[0])})
        return masks

    def sam_pair_embedding(self, masks1=None, masks2=None):
        predictor = SamPredictor(self.sam)
        ####### generate mask embbedings
        predictor.set_image(self.im1)  # (200,200,3)
        emb1 = predictor.get_image_embedding().cpu().numpy()  # (1, 256, 64, 64)
        predictor.set_image(self.im2)
        emb2 = predictor.get_image_embedding().cpu().numpy()
        # print(emb1.shape, emb2.shape)
        self.m1_embs = []
        self.m2_embs = []
        for _m in masks1:
            tmp_m = _m['segmentation'].astype(np.uint8)
            tmp_m = cv2.resize(tmp_m, (64, 64), interpolation=cv2.INTER_NEAREST)
            tmp_m = tmp_m.astype(bool)  # (64,64)
            tmp_m = tmp_m[np.newaxis, np.newaxis, :, :]
            tmp_emb = emb1 * tmp_m

            tmp_emb = np.array(
                [np.mean(channel[channel != 0]) if np.any(channel != 0) else 0 for channel in tmp_emb[0]])

            self.m1_embs.append(tmp_emb)

        for _m in masks2:
            tmp_m = _m['segmentation'].astype(np.uint8)
            tmp_m = cv2.resize(tmp_m, (64, 64), interpolation=cv2.INTER_NEAREST)
            tmp_m = tmp_m.astype(bool)  # (64,64)
            tmp_m = tmp_m[np.newaxis, np.newaxis, :, :]
            tmp_emb = emb2 * tmp_m

            tmp_emb = np.array(
                [np.mean(channel[channel != 0]) if np.any(channel != 0) else 0 for channel in tmp_emb[0]])
            self.m2_embs.append(tmp_emb)
        # print(len(masks1), len(masks2))
        # print(len(m1_embs), len(m2_embs))

        # return m1_embs, m2_embs

    def _cosine_similarity(self,vec1, vec2):
        return 1 - cosine(vec1, vec2)

    def _pair_masks(self,index_pairs, masks1, masks2):
        masks1_new = []
        masks2_new = []
        for i, j in index_pairs:
            masks1_new.append(masks1[i])
            masks2_new.append(masks2[j])
        return masks1_new, masks2_new

    def embedding_pair(self,masks1,masks2):
        similarity_matrix = np.zeros((len(masks1), len(masks2)))
        for i, vec_a in enumerate(masks1):
            for j, vec_b in enumerate(masks2):
                similarity_matrix[i, j] = self._cosine_similarity(vec_a, vec_b)

        sim_matrix = similarity_matrix.copy()
        self.similarity_matrix = (sim_matrix-sim_matrix.min())/(sim_matrix.max()-sim_matrix.min())
        index_pairs = []
        for i in range(min(len(masks1), len(masks2))):
            max_sim_idx = np.unravel_index(np.argmax(similarity_matrix, axis=None), similarity_matrix.shape)
            index_pairs.append(max_sim_idx)
            # pairs.append((masks2[max_sim_idx[0]], masks2[max_sim_idx[1]]))
            similarity_matrix[max_sim_idx[0], :] = -1
            similarity_matrix[:, max_sim_idx[1]] = -1
        # now obtain pairs, index_pairs

        self.masks1_cor, self.masks2_cor = self._pair_masks(index_pairs,self.masks1,self.masks2)
        # return masks1_cor, masks2_cor
        if len(self.masks1_cor) > 0:
            self.masks1_cor.append({'segmentation': self._maskfuse(self.masks1_cor)})
            self.masks2_cor.append({'segmentation': self._maskfuse(self.masks2_cor)})
        else:
            print('corresponding match fail in this slice!!')
            self.masks1_cor.append({'segmentation': np.zeros_like(self.im1[:,:,0])})
            self.masks2_cor.append({'segmentation': np.zeros_like(self.im2[:,:,0])})

    def overlap_pair(self, masks1,masks2):
        self.masks1_cor = []
        self.masks2_cor = []
        k = 0
        for mask in masks1[:-1]:
            k += 1
            print('mask1 {} is finding corresponding region mask...'.format(k))
            m1 = mask['segmentation']
            a1 = mask['area']
            v1 = np.mean(np.expand_dims(m1, axis=-1) * self.im1)
            overlap = m1 * masks2[-1]['segmentation'].astype(np.int64)
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
                    m21 = masks2[top_two[0]-1]['segmentation']
                    m22 = masks2[top_two[1]-1]['segmentation']
                    a21 = masks2[top_two[0]-1]['area']
                    a22 = masks2[top_two[1]-1]['area']
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

class RoiMatching():
    def __init__(self,img1,img2,device='cpu', v_min=200, v_max= 7000, mode = 'embedding'):
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
        model = SamModel.from_pretrained("facebook/sam-vit-huge").to(self.device)
        processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
        inputs = processor(image, return_tensors="pt").to(self.device)
        image_embeddings = model.get_image_embeddings(inputs["pixel_values"])
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
        self.masks1, self.masks2 = batched_outputs[0], batched_outputs[1]
        # self.masks1 = self._sam_everything(self.img1)  # len(RM.masks1) 2; RM.masks1[0] dict; RM.masks1[0]['masks'] list
        # self.masks2 = self._sam_everything(self.img2)
        self.masks1 = self._mask_criteria(self.masks1['masks'], v_min=self.v_min, v_max=self.v_max)
        self.masks2 = self._mask_criteria(self.masks2['masks'], v_min=self.v_min, v_max=self.v_max)

        match self.mode:
            case 'embedding':
                if len(self.masks1) > 0 and len(self.masks2) > 0:
                    self.embs1 = self._roi_proto(self.img1,self.masks1) #device:cuda1
                    self.embs2 = self._roi_proto(self.img2,self.masks2)
                    self.sim_matrix = self._similarity_matrix(self.embs1, self.embs2)
                    self.masks1, self.masks2 = self._roi_match(self.sim_matrix,self.masks1,self.masks2)
            case 'overlaping':
                self._overlap_pair(self.masks1,self.masks2)

class PairMode(PairMasks):
    def __init__(self, sam, im1, im2, mode = 'embedding'):
        self.sam = sam
        self.im1 = self.img_preprocess(im1)
        self.im2 = self.img_preprocess(im2)
        self.compare_mask_select()
        if mode == 'embedding':
            # self.sam_pair_embedding(self.masks1[:-1], self.masks2[:-1])
            self.embedding_pair(self.m1_embs, self.m2_embs, self.sm)
        elif mode == 'overlaping':
            self.overlap_pair(self.masks1, self.masks2)
        else:
            self.masks1_cor = None
            self.masks2_cor = None
    def set_sam(self,img,pred_iou_thresh=0.90, stability_score_thresh=0.9):
        mask_generator = SamAutomaticMaskGenerator(model=self.sam,
                                                   pred_iou_thresh=pred_iou_thresh,
                                                   # min_mask_region_area=150
                                                   # points_per_side=32
                                                   stability_score_thresh=stability_score_thresh,
                                                   )
        masks = mask_generator.generate(img)
        return masks
    def set_mask_filter(self, masks, v_min, v_max):
        masks = self._maskselect(masks, v_min, v_max)
        masks.append({'segmentation': self._maskfuse(masks)})
        return masks

    def sam_everything(self,img, pred_iou_thresh=0.90,stability_score_thresh=0.90,v_min=200, v_max=7000):
        masks = self.set_sam(img,pred_iou_thresh,stability_score_thresh)
        masks = self.set_mask_filter(masks, v_min, v_max)
        return masks

    def similarity_matrix(self, masks1, masks2): #masks is embs
        similarity_matrix = np.zeros((len(masks1), len(masks2)))
        for i, vec_a in enumerate(masks1):
            for j, vec_b in enumerate(masks2):
                similarity_matrix[i, j] = self._cosine_similarity(vec_a, vec_b)
        return similarity_matrix

    def sam_pair_embedding(self, masks1=None, masks2=None):
        predictor = SamPredictor(self.sam)
        ####### generate mask embbedings
        predictor.set_image(self.im1)  # (200,200,3)
        emb1 = predictor.get_image_embedding().cpu().numpy()  # (1, 256, 64, 64)
        predictor.set_image(self.im2)
        emb2 = predictor.get_image_embedding().cpu().numpy()
        # print(emb1.shape, emb2.shape)
        m1_embs = []
        m2_embs = []
        for _m in masks1:
            tmp_m = _m['segmentation'].astype(np.uint8)
            tmp_m = cv2.resize(tmp_m, (64, 64), interpolation=cv2.INTER_NEAREST)
            tmp_m = tmp_m.astype(bool)  # (64,64)
            tmp_m = tmp_m[np.newaxis, np.newaxis, :, :]
            tmp_emb = emb1 * tmp_m

            tmp_emb = np.array(
                [np.mean(channel[channel != 0]) if np.any(channel != 0) else 0 for channel in tmp_emb[0]])
            m1_embs.append(tmp_emb)

        for _m in masks2:
            tmp_m = _m['segmentation'].astype(np.uint8)
            tmp_m = cv2.resize(tmp_m, (64, 64), interpolation=cv2.INTER_NEAREST)
            tmp_m = tmp_m.astype(bool)  # (64,64)
            tmp_m = tmp_m[np.newaxis, np.newaxis, :, :]
            tmp_emb = emb2 * tmp_m

            tmp_emb = np.array(
                [np.mean(channel[channel != 0]) if np.any(channel != 0) else 0 for channel in tmp_emb[0]])
            m2_embs.append(tmp_emb)
        return m1_embs, m2_embs

    def compare_mask_select(self):
        masks1_1 = self.sam_everything(self.im1, v_min=200, v_max=7000, pred_iou_thresh=0.90,stability_score_thresh=0.90,)
        masks2_1 = self.sam_everything(self.im2, v_min=200, v_max=7000, pred_iou_thresh=0.90,stability_score_thresh=0.90,)
        masks1_2 = self.sam_everything(self.im1, v_min=200, v_max=20000, pred_iou_thresh=0.90,stability_score_thresh=0.80,)
        masks2_2 = self.sam_everything(self.im2, v_min=200, v_max=20000, pred_iou_thresh=0.90,stability_score_thresh=0.80,)
        embs1_1, embs2_1 = self.sam_pair_embedding(masks1_1[:-1], masks2_1[:-1])
        embs1_2, embs2_2 = self.sam_pair_embedding(masks1_2[:-1], masks2_2[:-1])
        s_m_1 = self.similarity_matrix(embs1_1, embs2_1)
        s_m_2 = self.similarity_matrix(embs1_2, embs2_2)
        if self.get_top_half(s_m_1) > self.get_top_half(s_m_2):
            self.m1_embs = embs1_1
            self.m2_embs = embs2_1
            self.sm = s_m_1
            self.masks1 = masks1_1
            self.masks2 = masks2_1
            print('mode1:7000')
        else:
            self.m1_embs = embs1_2
            self.m2_embs = embs2_2
            self.sm = s_m_2
            self.masks1 = masks1_2
            self.masks2 = masks2_2
            print('mode1:20000')

    def get_top_half(self, matrix):
        flattened_arr = matrix.flatten()
        sorted_arr = np.sort(flattened_arr)

        top_half_max_values = sorted_arr[-(matrix.size // 2):]

        return top_half_max_values.sum()


    def embedding_pair(self,masks1,masks2,similarity_matrix):
        '''
        masks1: list of binary masks generated from image1
        masks2: list of binary masks generated from image1
        similarity_matrix: cosine similarity between each mask1 and each mask2
        '''
        pairs = []
        index_pairs = []
        for i in range(min(len(masks1), len(masks2))):
            max_sim_idx = np.unravel_index(np.argmax(similarity_matrix, axis=None), similarity_matrix.shape)
            index_pairs.append(max_sim_idx)
            pairs.append((masks2[max_sim_idx[0]], masks2[max_sim_idx[1]]))

            # elminate the paired idx
            similarity_matrix[max_sim_idx[0], :] = -1
            similarity_matrix[:, max_sim_idx[1]] = -1
        # now obtain pairs, index_pairs

        self.masks1_cor, self.masks2_cor = self._pair_masks(index_pairs,self.masks1,self.masks2)
        # return masks1_cor, masks2_cor

class Vis():
    def __init__(self):
        pass

    def generate_distinct_colors(self,n):
        colors = []
        for i in range(n):
            # using HSV space
            hue = i / n
            saturation = 0.9
            value = 0.9
            rgb_color = mcolors.hsv_to_rgb([hue, saturation, value])
            colors.append(rgb_color)
        return colors

    def show_mask(self, mask, ax, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.4])], axis=0)
        else:
            color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image.astype('uint8'))

    def save_mask(self, mask, ax, color):
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(['none', color])
        ax.imshow(mask, cmap=cmap, alpha=0.5, interpolation='none')

    def show_points(self, coords, labels, ax, marker_size=375):
        coords = np.array(coords)
        labels = np.array(labels)
        pos_points = coords[labels == 1]
        neg_points = coords[labels == 0]
        # print("neg_points: ",neg_points)
        # ax.scatter([_r[0] for _r in pos_points], [_r[1] for _r in pos_points], color='green', marker='*', s=marker_size, edgecolor='white',
        #            linewidth=1.25)
        # ax.scatter([_r[0] for _r in neg_points], [_r[1] for _r in neg_points], color='red', marker='*', s=marker_size, edgecolor='white',
        #            linewidth=1.25)
        ax.scatter(pos_points[:, 1], pos_points[:, 0], color='green', marker='*', s=marker_size, edgecolor='white',
                   linewidth=1.25)
        ax.scatter(neg_points[:, 1], neg_points[:, 0], color='red', marker='*', s=marker_size, edgecolor='white',
                   linewidth=1.25)

    def show_box(self, box, ax):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

    def _show_img(self, image, masks, input_point=None, input_label=None, ind=1):
        plt.figure(ind)
        plt.imshow(image)
        for mask in masks[:-1]:
            # masks = MaskSelect(masks)
            # for mask in masks:
            # show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
            # print(mask['segmentation'].shape,mask['segmentation'].sum(),mask['segmentation'].max())
            self.show_mask(mask['segmentation'], plt.gca(), random_color=True)
            # show_mask(mask, plt.gca(), random_color=True)
        if input_point is not None:
            for points, labels in zip(input_point, input_label):
                self.show_points(points, labels, plt.gca())
        plt.axis('off')
        # plt.show()
    def _show_predict_img(self, image1, masks1, image2, masks2, n,m, vis_id):
        plt.figure(2*vis_id-1)
        # plt.imshow(image1)
        # self.show_mask(masks1, plt.gca(), random_color=False)
        for i in range(n):
            plt.subplot(1,n,i+1)
            plt.imshow(image1)
            self.show_mask(masks1[i], plt.gca(), random_color=False)
        plt.axis('off')

        plt.figure(2*vis_id)
        for i in range(m):
            plt.subplot(1,m,i+1)
            plt.imshow(image2)
            self.show_mask(masks2[i], plt.gca(), random_color=False)
        plt.axis('off')



    def _show_cor_img(self, image1, image2, masks1, masks2):
        colors = self.generate_distinct_colors(20)
        for ind in range(len(masks1)):
            plt.figure(ind)
            color = np.random.rand(3,)

            plt.subplot(1, 2, 1)
            plt.imshow(image1.astype('uint8'))
            # self.show_mask(masks1[ind]['segmentation'], plt.gca(), random_color=True)
            self.save_mask(masks1[ind]['segmentation'], plt.gca(), color=colors[ind])
            plt.axis('off')


            plt.subplot(1, 2, 2)
            plt.imshow(image2.astype('uint8'))
            # self.show_mask(masks2[ind]['segmentation'], plt.gca(), random_color=True)
            self.save_mask(masks2[ind]['segmentation'], plt.gca(), color=colors[ind])
            plt.axis('off')


def load_nii_data(path):
    import monai
    from monai.transforms import Compose, LoadImaged, Spacingd, ToTensord, Resized, AddChanneld
    transforms = Compose([
        LoadImaged(keys=["image"]),
        AddChanneld(keys=["image"]),
        Resized(keys=["image"], spatial_size=(96, 200, 200)),
        ToTensord(keys=["image"])
    ])
    dataset = monai.data.Dataset(data=[{"image": path}], transform=transforms)
    sample = dataset[0]
    image_tensor = sample["image"]
    image_tensor = image_tensor.detach().numpy()
    image_tensor = image_tensor[0]

    return image_tensor

def expand_3c(slice_2d):
    slice_2d = (slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min() + 1e-5)
    slice_2d *= 255.
    slice_2d = np.uint8(slice_2d)
    # print(slice_2d.shape)

    slice_3_channel = np.dstack((slice_2d, slice_2d, slice_2d))
    return slice_3_channel

def slice_pair(data1,data2,ind):
    assert ind <= data1.shape[0]

    slice1 = data1[ind]
    slice2 = data2[ind]
    slice1 = expand_3c(slice1)
    slice2 = expand_3c(slice2)

    return slice1, slice2

