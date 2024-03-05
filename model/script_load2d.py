import os
from dataset.datasets import load_data_volume
import argparse
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
import numpy as np
import logging
import torch.nn.functional as F

from monai.losses import DiceCELoss, DiceLoss
from monai.inferers import sliding_window_inference

# import surface_distance
# from surface_distance import metrics

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
# from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor

import numpy as np
from scipy import ndimage
from skimage import exposure
import cv2
from scipy.spatial.distance import cosine

device = "cuda"
# device = 'cpu'

def image_histogram_equalization(image, number_bins=256):
    # 计算图像的直方图
    image_histogram, bins = np.histogram(image.flatten(), number_bins, density=True)
    cdf = image_histogram.cumsum()  # 累计分布函数
    cdf = (number_bins - 1) * cdf / cdf[-1]  # 归一化
    equalized_image = np.interp(image.flatten(), bins[:-1], cdf)
    return equalized_image.reshape(image.shape)


def apply_clahe(image, clip_limit=0.01, tile_grid_size=(8, 8)):
    # 分割成小块
    tiles = [np.hsplit(row, tile_grid_size[1]) for row in np.vsplit(image, tile_grid_size[0])]
    clahe_tiles = []

    for row in tiles:
        clahe_row = []
        for tile in row:
            # 对每个小块应用直方图均衡化
            tile_hist_eq = image_histogram_equalization(tile)

            # 限制对比度
            if clip_limit is not None:
                # 计算clip限制
                clip_limit_abs = np.int32(clip_limit * tile.size)
                tile_hist_eq = exposure.rescale_intensity(tile_hist_eq, in_range=(
                np.percentile(tile_hist_eq, clip_limit_abs), np.percentile(tile_hist_eq, 100 - clip_limit_abs)))

            clahe_row.append(tile_hist_eq)
        clahe_tiles.append(clahe_row)

    # 重新组合小块
    clahe_image = np.vstack([np.hstack(row) for row in clahe_tiles])
    return clahe_image

def norm(img):
    return (img-img.min())/(img.max()-img.min())

def generate_2d(test_data):
    for idx, (img, seg, spacing) in enumerate(test_data):
        if idx == 3:
            img = img.float()
            img = img[:, :1, :, :, :]
            img = F.interpolate(img, size=seg.shape[1:], mode="trilinear")
            s = img.shape  # (1,96,200,200)
            # print(s)
            img = img.detach().numpy()

            # print(img.max(),img.min())
            # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img = norm(img)
            # print(img.max(),img.min())
            img = img * 255.

            slice_1 = img[0, 0, 45, :, :]
            slice_2 = img[0, 0, :, 100, :]
            cv2.imwrite(os.path.join(path, 'slice_1.png'), slice_1)
            cv2.imwrite(os.path.join(path, 'slice_2.png'), slice_2)
            break




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", default="prostate", type=str, choices=["kits", "pancreas", "lits", "colon","prostate",'miami_prostate']
    )
    parser.add_argument(
        "--snapshot_path",
        default="/raid/shiqi/Results_tl",
        type=str,
    )
    parser.add_argument(
        "--data_prefix",
        default="/raid/shiqi/data/Data",
        type=str,
    )
    parser.add_argument(
        "--data_split_prefix",
        default="../datafile",
        type=str,
    )
    parser.add_argument(
        "-m",
        "--method",
        default="unetr",
        type=str,
        choices=["swin_unetr", "unetr", "3d_uxnet", "nnformer", "unetr++", "transbts", "unetr_2"],
    )
    parser.add_argument(
        "-t",
        "--task",
        default="lesion",
        type=str,
        choices=["prostate_mask", "lesion"],
    )
    parser.add_argument("--overlap", default=0.7, type=float)
    parser.add_argument(
        "--infer_mode", default="constant", type=str, choices=["constant", "gaussian"]
    )
    parser.add_argument("--save_pred", action="store_true")
    parser.add_argument("--num_classes", default=2, type=int)
    parser.add_argument("--num_worker", default=6, type=int)
    parser.add_argument("--tolerance", default=5, type=int)
    parser.add_argument("--num_gpus", default=-1, type=int)


    args = parser.parse_args()
    test_data = load_data_volume(
        data=args.data,
        batch_size=1,
        task=args.task,
        path_prefix=args.data_prefix,
        split_prefix=args.data_split_prefix,
        augmentation=False,
        split="test",
        convert_to_sam=False,
        num_worker=args.num_worker,
        deterministic=True,
    )
    path = r'/raid/shiqi/'
    import cv2
    generate_2d()

def stack():
    path = r'/raid/shiqi/'
    import cv2
    img1 = cv2.imread(os.path.join(path,'slice_2_1.png'))
    img2 = cv2.imread(os.path.join(path,'slice_2_2.png'))
    img = np.hstack([img1,img2])
    cv2.imwrite(os.path.join(path, 'slice_2_12.png'), img)

def color():
    path = r'/raid/shiqi/'
    import cv2
    grayscale_image = cv2.imread(os.path.join(path,'slice_1_12.png'),cv2.IMREAD_GRAYSCALE)
    color_image = cv2.applyColorMap(grayscale_image, cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(path, 'slice_1_12_c.png'), color_image)

# from pycocotools import mask as mask_utils
import lzstring
import base64
def process_image(img,predictor):
    predictor.set_image(img)

    image_embedding = predictor.get_image_embedding().cpu().numpy()

    # result_base64 = base64.b64encode(image_embedding.tobytes()).decode('utf-8')
    # result_list = [result_base64]
    return image_embedding

def automatic_masks(img):
    sam = sam_model_registry["vit_h"](checkpoint="/raid/shiqi/sam_pretrained/sam_vit_h_4b8939.pth")
    sam.to(device=device)
    predictor = SamPredictor(sam)
    mask_generator = SamAutomaticMaskGenerator(sam)

    mask = mask_generator.generate(img)

    sorted_anns = sorted(mask, key=(lambda x: x['area']), reverse=True)

    lzs = lzstring.LZString()

    res = []
    for ann in sorted_anns:
        m = ann['segmentation']

        source_mask = mask_utils.encode(m)['counts']

        # encoded = lzs.compressToEncodedURIComponent(source_mask)

        r = {
            "encodedMask": source_mask,
            "point_coord": ann['point_coords'][0],
        }
        res.append(r)
    process_image(img,predictor)
    return res

def sam_mask_reg(grayscale_image):
    # path = r'/raid/shiqi/'
    # import cv2
    # grayscale_image = cv2.imread(os.path.join(path, 'slice_1_2.png'))

    # from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
    sam = sam_model_registry["vit_h"](checkpoint="/raid/shiqi/sam_pretrained/sam_vit_h_4b8939.pth")
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(model=sam,
                                               pred_iou_thresh=0.90,
                                               # min_mask_region_area=150
                                               # points_per_side=32
                                               stability_score_thresh=0.9,
                                               )
    mask_generator_default = SamAutomaticMaskGenerator(model=sam )
    masks = mask_generator.generate( grayscale_image)
    masks_default = mask_generator_default.generate(grayscale_image)
    # for _m in masks:
    #     print(_m['area'])

    # sam输入：必须是3通道
    # sam输出：mask is a list and its len is 19.
    # p masks[0].keys()
    # dict_keys(['segmentation', 'area', 'bbox', 'predicted_iou', 'point_coords', 'stability_score', 'crop_box'])
    # p masks[0]['segmentation'].shape
    # (200, 400) #numpy
    # p masks[0]['area']
    # 557
    #  p masks[0]['bbox']
    # [0, 0, 31, 29]
    # p masks[0]['predicted_iou']
    # 0.9894084930419922
    # (Pdb) p masks[0]['point_coords']
    # [[18.75, 9.375]]
    # (Pdb) p masks[0]['stability_score']
    # 0.987522304058075
    # (Pdb) p masks[0]['crop_box']
    # [0, 0, 400, 200]
    # (Pdb) p grayscale_image.shape
    # (200, 400, 3)
    # for i in range(len(masks)):
    #     s = masks[i]['bbox'][2:]
    #     cv2.imwrite(os.path.join(path, 'slice_1_2_s_{}_{}.png'.format(str(i),str(s))), np.uint8(masks[i]['segmentation'])*255.)
    # masks = np.vstack(masks,np.ones((masks.shape[-2],masks.shape[-1])))
    return masks, masks_default

def sam_prompt_reg(grayscale_image):
    # path = r'/raid/shiqi/'
    # import cv2
    # grayscale_image = cv2.imread(os.path.join(path, 'slice_1_2.png'))

    # from segment_anything import SamPredictor, sam_model_registry
    sam = sam_model_registry["vit_h"](checkpoint="/raid/shiqi/sam_pretrained/sam_vit_h_4b8939.pth")
    sam.to(device=device)
    predictor = SamPredictor(sam)
    predictor.set_image(grayscale_image)
    # point_coords = np.array([[204, 2],[11,8],[190,2],[394,2],[100,19],[298,8]])
    # point_labels = np.array([1,1,1,1,2,2])
    point_coords = np.array([[12,94]])
    point_labels = np.array([1])
    embedding = predictor.get_image_embedding().cpu().numpy()
    masks, scores, logits = predictor.predict(point_coords=point_coords,point_labels=point_labels)
    # masks = np.vstack(masks, np.ones((masks.shape[-2], masks.shape[-1])))
    # print(np.argmax(scores)) # 0(第一张是最好的）
    # masks(MaskFuse(masks))
    # masks = masks[-2:]

    # for i in range(len(masks)):
        # s = masks[i]['bbox']
        # cv2.imwrite(os.path.join(path, 'slice_1_2_sp_{}_{}.png'.format(str(i),str(scores[i]))), np.uint8(masks[i])*255.)
    return masks,scores,logits,embedding

def mutual_seg(im1,im2,masks_1,masks_2):
    # masks:[0,1..]+[multiclass]
    # masks[0]['segmentation']
    masks1 = masks_1[:]
    masks2 = masks_2[:]
    im1 = im1
    im2 = im2
    total_coords = []
    total_labels = []
    for i in range(len(masks2)-1):
        tmp_m2 = masks2[i]['segmentation']
        r = (tmp_m2*(masks1[-1]['segmentation']>0)).sum()/tmp_m2.sum()
        # print(r)
        if r < 0.4:
            all_cover = False
            for _m1 in masks1[:-1]:
                a = (tmp_m2*_m1['segmentation']).sum()/_m1['segmentation'].sum()
                if a > 0.85:
                    all_cover = True
                    break
            if not all_cover:
                print('======================r<0.15 with mask {}th======================'.format(i))
                point_x = masks2[i]['bbox'][0]+round(masks2[i]['bbox'][2]/2)
                point_y = masks2[i]['bbox'][1]+round(masks2[i]['bbox'][3]/2)
                positive_point = np.array([[point_x,point_y]])

        #      add the positive_point to im1 to generate a new mask1_p
        #         from segment_anything import SamPredictor, sam_model_registry
                sam = sam_model_registry["vit_b"](checkpoint="/raid/shiqi/sam_pretrained/sam_vit_b_01ec64.pth")
                sam.to(device=device)
                predictor = SamPredictor(sam)
                predictor.set_image(im1)
                point_coords =positive_point
                point_labels = np.array([1])
                masks1_p, scores, logits = predictor.predict(point_coords=point_coords, point_labels=point_labels)

            #     add negative point to mask1_p to elimitate overcover area
                # print(masks1_p[0].shape,masks1[-1]['segmentation'])
                masks1_overlap = masks1_p[0] * masks1[-1]['segmentation']
                # print("overlap.shape:",masks1_overlap.shape)
                if masks1_overlap.sum() > 0:
                    negative_point_x = []
                    negative_point_y = []
                    for _i in np.unique(masks1_overlap):
                        if _i == 0: continue
                        indices = np.argwhere(masks1_overlap == _i)
                        # print('indice:',indices)
                        chosen_index = indices[np.random.choice(len(indices))]
                        negative_point_x.append(chosen_index[0])
                        negative_point_y.append(chosen_index[1])
                    for p in range(len(negative_point_x)):
                        tmp_point = []
                        tmp_point.append(negative_point_x[p])
                        tmp_point.append(negative_point_y[p])
                        point_coords = np.vstack((point_coords, tmp_point))
                        point_labels = np.append(point_labels, 0)
                    mask_input = logits[np.argmax(scores), :, :]
                    point_coords = point_coords
                    # print('point_coords:',point_coords)
                    point_labels = point_labels
                    total_coords.append(point_coords.tolist())
                    total_labels.append(point_labels.tolist())

                    masks1_p, _, _ = predictor.predict(
                        point_coords=point_coords,
                        point_labels=point_labels,
                        mask_input=mask_input[None, :,:],
                        multimask_output=False,
                    )
                    masks1.pop(-1)
                    masks1.append({'segmentation':masks1_p[0,:,:]})
                    masks1.append({'segmentation':MaskFuse(masks1[:len(masks1)])})
    return masks1,total_coords,total_labels

def corresponding_mask(im1,im2,masks1,masks2):
    # print('masks1: {}'.format(len(masks1)))
    # print('masks2: {}'.format(len(masks2)))
    masks1_new = []
    masks2_new = []
    k = 0
    for mask in masks1[:-1]:
        k += 1
        print('mask1 {} is finding corresponding region mask...'.format(k))
        m1 = mask['segmentation']
        a1 = mask['area']
        v1 = np.mean(np.expand_dims(m1, axis=-1)*im1)
        overlap = m1*masks2[-1]['segmentation'].astype(np.int64)
        # print(np.unique(overlap))
        if (overlap>0).sum()/a1 > 0.3:
            counts = np.bincount(overlap.flatten())
            # print(counts)
            sorted_indices = np.argsort(counts)[::-1]
            top_two = sorted_indices[1:3]
            # print(top_two)
            if top_two[-1] == 0:
                cor_ind = 0
            elif abs(counts[top_two[-1]]-counts[top_two[0]])/max(counts[top_two[-1]],counts[top_two[0]]) < 0.2:
                cor_ind = 0
            else:
                cor_ind = 0
                # m21 = masks2[top_two[0]-1]['segmentation']
                # m22 = masks2[top_two[1]-1]['segmentation']
                # a21 = masks2[top_two[0]-1]['area']
                # a22 = masks2[top_two[1]-1]['area']
                # v21 = np.mean(np.expand_dims(m21, axis=-1)*im2)
                # v22 = np.mean(np.expand_dims(m22, axis=-1)*im2)
                # if np.abs(a21-a1) > np.abs(a22-a1):
                #     cor_ind = 0
                # else:
                #     cor_ind = 1
                # print('area judge to cor_ind {}'.format(cor_ind))
                # if np.abs(v21-v1) < np.abs(v22-v1):
                #     cor_ind = 0
                # else:
                #     cor_ind = 1
                # print('value judge to cor_ind {}'.format(cor_ind))
            print('mask1 {} has found the corresponding region mask: mask2 {}'.format(k,top_two[cor_ind]))

            masks2_new.append(masks2[top_two[cor_ind]-1])
            masks1_new.append(mask)
    return masks1_new, masks2_new

def generate_emb(sam,image1, image2, masks1=None, masks2=None):

    predictor = SamPredictor(sam)
    ####### generate mask embbedings
    predictor.set_image(image1) # (200,200,3)
    emb1 = predictor.get_image_embedding().cpu().numpy() # (1, 256, 64, 64)
    predictor.set_image(image2)
    emb2 = predictor.get_image_embedding().cpu().numpy()
    # print(emb1.shape, emb2.shape)
    m1_embs = []
    m2_embs = []
    for _m in masks1:
        tmp_m = _m['segmentation'].astype(np.uint8)
        tmp_m = cv2.resize(tmp_m, (64, 64), interpolation=cv2.INTER_NEAREST)
        tmp_m = tmp_m.astype(bool) #(64,64)
        tmp_m = tmp_m[np.newaxis, np.newaxis,:,:]
        tmp_emb = emb1 * tmp_m

        tmp_emb = np.array([np.mean(channel[channel != 0]) if np.any(channel != 0) else 0 for channel in tmp_emb[0]])
        m1_embs.append(tmp_emb)

    for _m in masks2:
        tmp_m = _m['segmentation'].astype(np.uint8)
        tmp_m = cv2.resize(tmp_m, (64, 64), interpolation=cv2.INTER_NEAREST)
        tmp_m = tmp_m.astype(bool) #(64,64)
        tmp_m = tmp_m[np.newaxis, np.newaxis,:,:]
        tmp_emb = emb2 * tmp_m

        tmp_emb = np.array([np.mean(channel[channel != 0]) if np.any(channel != 0) else 0 for channel in tmp_emb[0]])
        m2_embs.append(tmp_emb)
    print(len(masks1),len(masks2))
    print(len(m1_embs),len(m2_embs))

    return m1_embs,m2_embs

def cosine_similarity(vec1, vec2):
    return 1 - cosine(vec1, vec2)

def pair_emb(A,B):
    # 计算相似度矩阵
    similarity_matrix = np.zeros((len(A), len(B)))
    for i, vec_a in enumerate(A):
        for j, vec_b in enumerate(B):
            similarity_matrix[i, j] = cosine_similarity(vec_a, vec_b)
    # 寻找最大相似度的配对
    pairs = []
    index_pairs = []
    for i in range(min(len(A), len(B))):
        # 找到最大相似度的索引
        max_sim_idx = np.unravel_index(np.argmax(similarity_matrix, axis=None), similarity_matrix.shape)
        index_pairs.append(max_sim_idx)
        pairs.append((A[max_sim_idx[0]], B[max_sim_idx[1]]))

        # 将已配对的向量从后续考虑中排除
        similarity_matrix[max_sim_idx[0], :] = -1
        similarity_matrix[:, max_sim_idx[1]] = -1
    return pairs, index_pairs

def pair_masks(index_pairs, masks1,masks2):
    masks1_new = []
    masks2_new = []
    for i,j in index_pairs:
        masks1_new.append(masks1[i])
        masks2_new.append(masks2[j])
    return masks1_new,masks2_new




def MaskFuse(masks):
    img = np.zeros(masks[0]['segmentation'].shape)
    for i in range(len(masks)):
        _m = np.uint8(masks[i]['segmentation'])
        # if (_m * img).sum() > 0:
        #     print('overlap!')
        if (_m*img).sum() > 0:
            _m = _m-np.uint8(_m*(img>0))
        #
        #     if (img+_m_n*(i+1)).max() > len(masks):
        #         import pdb
        #         pdb.set_trace()
        img += _m*(i+1)
        if img.max()> len(masks):
            import pdb
            pdb.set_trace()
    return img

def MaskSelect(masks, v_min=200, v_max=7000):
    remove_list = set()
    for _i,mask in enumerate(masks):
        # print(mask['area'])
        if mask['segmentation'].sum() < v_min or mask['segmentation'].sum() > v_max:
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

def sam_everything(sam,img):

    # from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    # sam = sam_model_registry["vit_h"](checkpoint="/raid/shiqi/sam_pretrained/sam_vit_h_4b8939.pth")
    # sam.to(device=device)
    # mask_generator = SamAutomaticMaskGenerator(model=sam,
    #                                         # points_per_side=64,#控制采样点的间隔，值越小，采样点越密集
    #                                         # pred_iou_thresh=0.86,#mask的iou阈值
    #                                         # stability_score_thresh=0.92,#mask的稳定性阈值
    #                                         # crop_n_layers=1,
    #                                         # crop_n_points_downscale_factor=2,
    #                                         min_mask_region_area=5, #最小mask面积，会使用opencv滤除掉小面积的区域
    #                                            )
    mask_generator = SamAutomaticMaskGenerator(model=sam,
                                               pred_iou_thresh=0.90,
                                               # min_mask_region_area=150
                                               # points_per_side=32
                                               stability_score_thresh=0.9,
                                               )
    masks = mask_generator.generate(img)
    # import pdb
    # pdb.set_trace()
    masks = MaskSelect(masks)
    masks.append({'segmentation':MaskFuse(masks)})
    return masks

def unet_everything(model,img):
    mask_generator = SamAutomaticMaskGenerator(model=model,
                                               pred_iou_thresh=0.90,
                                               # min_mask_region_area=150
                                               # points_per_side=32
                                               stability_score_thresh=0.9,
                                               )
    masks = mask_generator.generate(img)
    # import pdb
    # pdb.set_trace()
    # masks = MaskSelect(masks)
    masks.append({'segmentation': MaskFuse(masks)})
    return masks

def _transition(img1,img2):
    transition_width = 20
    transition = np.zeros((img1.shape[0], transition_width, 3))
    for i in range(transition_width):
        weight = i / transition_width
        transition[:, i, :] = (1 - weight) * img1[:, -1, :] + weight * img2[:, 0, :]
    result = np.hstack((img1, transition, img2))
    return np.uint8(result)

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.4])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
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


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

def _show_img(image,masks,input_point=None, input_label=None,ind=1):
    plt.figure(ind)
    plt.imshow(image)
    for mask in masks[:-1]:
    # masks = MaskSelect(masks)
    # for mask in masks:
        # show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
        # print(mask['segmentation'].shape,mask['segmentation'].sum(),mask['segmentation'].max())
        show_mask(mask['segmentation'], plt.gca(), random_color=True)
        # show_mask(mask, plt.gca(), random_color=True)
    if input_point is not None:
        for points,labels in zip(input_point,input_label):
            show_points(points, labels, plt.gca())
    plt.axis('off')
    # plt.show()

def _show_cor_img(image1,image2, masks1, masks2):
    for ind in range(len(masks1)):
        plt.figure(ind)

        plt.subplot(1, 2, 1)
        plt.imshow(image1)
        show_mask(masks1[ind]['segmentation'], plt.gca(), random_color=True)

        plt.subplot(1, 2, 2)
        plt.imshow(image2)
        show_mask(masks2[ind]['segmentation'], plt.gca(), random_color=True)
        plt.axis('off')











if __name__ == '__main__':
    # main()
    # stack()
    # color()
    # sam_prompt_reg()
    path = r'/raid/shiqi/'
    # import cv2

    ######## Sam Trial #####################
    sam = sam_model_registry["vit_h"](checkpoint="/raid/shiqi/sam_pretrained/sam_vit_h_4b8939.pth")
    sam.to(device=device)

    ########## UNet Trial #####################
    # from utils.model_util import get_model
    #
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "-m",
    #     "--method",
    #     default="unetr_2d",
    #     type=str,
    #     choices=["swin_unetr", "unetr", "3d_uxnet", "nnformer", "unetr++", "transbts", "unetr_2", "unetr_2d"],
    # )
    # args = parser.parse_args()
    # seg_net = get_model(args)
    # seg_net.cuda()


    image1 = cv2.imread(os.path.join(path, 'slice_2_1.png'))
    image2 = cv2.imread(os.path.join(path, 'slice_2_3.png'))
    print(image1.shape)
    import pdb
    pdb.set_trace()
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    # image3 = _transition(image1,image2)
    image1 = torch.from_numpy(image1)
    print(image1.shape)

    masks = unet_everything(sam,image1)
    pdb.set_trace()
    _show_img(image1,masks)


    # masks,scores,logits, emb = sam_prompt_reg(image1)
    # _show_img(image1,[masks[0]])


    # masks, masks_default = sam_mask_reg(image1)
    # _show_img(image1,masks,ind=1)
    # _show_img(image1,masks_default,ind=2)

    # masks_1 = sam_everything(sam,image1) #7
    # masks_2 = sam_everything(sam,image2) #12

    # emb_1, emb_2 = generate_emb(sam,image1,image2,masks_1[:-1],masks_2[:-1])
    # pairs, pairs_index = pair_emb(emb_1,emb_2)
    # masks_1_cor, masks_2_cor = pair_masks(pairs_index, masks_1, masks_2)




    # plt.figure(1)
    # plt.imshow(image1)
    # show_mask(masks_1[3]['segmentation'],plt.gca(), random_color=False)
    # plt.figure(2)
    # plt.imshow(image1)
    # show_mask(masks_1[-1]['segmentation']==4,plt.gca(), random_color=False)

    # masks_1_cor, masks_2_cor = corresponding_mask(image1,image2,masks_1,masks_2,)
    # _show_cor_img(image1,image2,masks_1_cor,masks_2_cor,)

    # up_masks_1, points1, labels1 = mutual_seg(image1,image2,masks_1,masks_2)
    # up_masks_2, points2, labels2 = mutual_seg(image2,image1,masks_2,masks_1)

    # cv2.imwrite(os.path.join(path, 'mask_1.png'), np.uint8(masks_1[-1]['segmentation']/np.max(masks_1[-1]['segmentation']) * 255.))
    # cv2.imwrite(os.path.join(path, 'mask_2.png'), np.uint8(masks_2[-1]['segmentation']/np.max(masks_2[-1]['segmentation']) * 255.))
    # cv2.imwrite(os.path.join(path, 'mask_1_up.png'), np.uint8(up_masks_1[-1]['segmentation']/np.max(up_masks_1[-1]['segmentation']) * 255.))
    # cv2.imwrite(os.path.join(path, 'mask_2_up.png'), np.uint8(up_masks_2[-1]['segmentation']/np.max(up_masks_2[-1]['segmentation']) * 255.))

    # _show_img(image1, up_masks_1, points1, labels1,ind=1)
    # _show_img(image2, up_masks_2, points2, labels2,ind=2)
    # _show_img(image1, masks_1, ind=3)
    # _show_img(image2, masks_2, ind=4)
    plt.show()

