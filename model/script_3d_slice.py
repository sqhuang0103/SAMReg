from pair2d import PairMasks, Vis, PairMode
import os
from .dataset.datasets import load_data_volume
import argparse
import matplotlib.pyplot as plt
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
from scipy.spatial.distance import cosine
from monai.transforms import (
    Compose,
    RandCropByPosNegLabel,
    RandCropByPosNegLabeld,
    CropForegroundd,
    SpatialPadd,
    ScaleIntensityRanged,
    MapTransform,)
from monai.config import IndexSelection, KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from scipy import ndimage



import torch
import torch.nn as nn
import numpy as np
import logging
from utils.model_util import get_model
import torch.nn.functional as F
from typing import Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Union
from scipy.fftpack import fft2, ifft2
from script_3d import sam_slice_v2, sam_slice

class BinarizeLabeld(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        threshold: float = 0.5,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.threshold = threshold

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            if not isinstance(d[key], torch.Tensor):
                d[key] = torch.as_tensor(d[key])

            dtype = d[key].dtype
            d[key] = (d[key] > self.threshold).to(dtype)
        return d

def norm(img):
    return (img-img.min())/(img.max()-img.min())

def expand_3c(slice_2d):
    slice_2d = (slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min() + 1e-5)
    slice_2d *= 255.
    slice_2d = np.uint8(slice_2d)
    # print(slice_2d.shape)

    # 将切片扩展为 3 通道，形状为 (200, 200,3)
    slice_3_channel = np.dstack((slice_2d, slice_2d, slice_2d))
    return slice_3_channel

def reassign_segmentation_labels(segmentation, overlap_threshold=0.8):
    depth, height, width = segmentation.shape
    new_labels = np.zeros_like(segmentation)
    label_counter = 1
    label_mapping = {}

    for i in range(depth):
        current_plane = segmentation[i]

        # 标记当前切片的连通区域
        labeled_array, num_features = ndimage.label(current_plane)

        # 遍历每个连通区域
        for region_label in range(1, num_features + 1):
            region_mask = labeled_array == region_label

            # 如果这是序列中的第一个切片，或者区域在前一个切片中不存在
            if i == 0 or not np.any(new_labels[i - 1][region_mask]):
                new_labels[i][region_mask] = label_counter
                label_mapping[(i, region_label)] = label_counter
                label_counter += 1
            else:
                # 检查这个区域是否与前一个切片中的区域重叠
                overlapping_labels = np.unique(new_labels[i - 1][region_mask])
                overlapping_labels = overlapping_labels[overlapping_labels != 0]

                if len(overlapping_labels) > 0:
                    # 如果有重叠，使用相同的标签
                    new_labels[i][region_mask] = overlapping_labels[0]
                    label_mapping[(i, region_label)] = overlapping_labels[0]
                else:
                    # 否则，分配新的标签
                    new_labels[i][region_mask] = label_counter
                    label_mapping[(i, region_label)] = label_counter
                    label_counter += 1

    return new_labels

def _nii_seg(test_data,sam,data_idx=5):
    from script_3d import write_nii_data,sam_slice,_show_image
    mask_generator = SamAutomaticMaskGenerator(model=sam,
                                               pred_iou_thresh=0.90,
                                               # min_mask_region_area=150
                                               # points_per_side=32
                                               stability_score_thresh=0.9,
                                               )
    for idx, (img, seg, spacing) in enumerate(test_data):
        if idx == data_idx:
            img = img.float()
            img = img[:, :1, :, :, :]
            img = F.interpolate(img, size=seg.shape[1:], mode="trilinear")
            # s = img.shape  # (1,96,200,200)
            # print(s)
            img = img.detach().numpy()

            # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img = norm(img)
            # print(img.max(),img.min())
            img = img * 255.
            # print(img.shape) #(1,1,96,200,200)
            img = img[0] # (1, 96, 200, 200)
            img = img[0]
            # img = fre_filter(img)
            # img = gau_filter(img)

            result = sam_slice_v2(mask_generator, img, axis=0)
            # result = reassign_segmentation_labels(result)
            write_nii_data('/raid/shiqi', 'test_id{}_a0_img.nii.gz'.format(data_idx), img)
            write_nii_data('/raid/shiqi', 'test_id{}_a0_f.nii.gz'.format(data_idx), result)
            # _show_image(img, result, 10, 1)
            # _show_image(img, result, 45, 2)
            # _show_image(img, result, 80, 3)
            # plt.show()


            break
    return result

def _nii_seg_multi_axis(test_data,sam):
    from script_3d import write_nii_data,sam_slice
    mask_generator = SamAutomaticMaskGenerator(model=sam,
                                               pred_iou_thresh=0.90,
                                               # min_mask_region_area=150
                                               # points_per_side=32
                                               stability_score_thresh=0.9,
                                               )
    for idx, (img, seg, spacing) in enumerate(test_data):
        if idx == 5:
            img = img.float()
            img = img[:, :1, :, :, :]
            img = F.interpolate(img, size=seg.shape[1:], mode="trilinear")
            # s = img.shape  # (1,96,200,200)
            # print(s)
            img = img.detach().numpy()

            # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img = norm(img)
            # print(img.max(),img.min())
            img = img * 255.
            # print(img.shape) #(1,1,96,200,200)
            img = img[0]
            img = img[0]
            # img = fre_filter(img)


            result_a0 = sam_slice(mask_generator, img, axis=0)
            result_3d = np.zeros_like(result_a0)
            result_3d += result_a0
            result_a1 = sam_slice(mask_generator, img, axis=1)
            result_3d += result_a1
            result_a2 = sam_slice(mask_generator, img, axis=2)
            result_3d += result_a2
            result_3d = np.uint8(result_3d>0)
            write_nii_data('/raid/shiqi/', 'test_id5_a0_f.nii.gz', result_a0)
            write_nii_data('/raid/shiqi/', 'test_id5_a1_f.nii.gz', result_a1)
            write_nii_data('/raid/shiqi/', 'test_id5_a2_f.nii.gz', result_a2)
            write_nii_data('/raid/shiqi/', 'test_id5_3d_f.nii.gz', result_3d)
            break



def slice_pair(test_data, ind1, ind2, slice_num):
    slice_list = []
    for idx, (img, seg, spacing) in enumerate(test_data):
        if idx == ind1 or idx == ind2:
            img = img.float()
            img = img[:, :1, :, :, :]
            img = F.interpolate(img, size=seg.shape[1:], mode="trilinear")
            s = img.shape  # (1,96,200,200)
            # print(s)
            img = img.detach().numpy()

            # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img = norm(img)
            # print(img.max(),img.min())
            img = img * 255.
            slice_2d = img[0, 0, slice_num, :, :]
            slice_2d = np.uint8(slice_2d)
            slice_3d = np.dstack((slice_2d, slice_2d, slice_2d))
            slice_list.append(slice_3d)

            if len(slice_list) == 2:
                break
    return slice_list

def _slice_prompt(test_data, sam, data_idx = 5,slice1_idx=70,slice2_idx=69):

    # load near slices
    for idx, (img, seg, spacing) in enumerate(test_data):
        if idx == data_idx:
            img = img.float()
            img = img[:, :1, :, :, :]
            img = F.interpolate(img, size=seg.shape[1:], mode="trilinear")
            s = img.shape  # (1,96,200,200)
            # print(s)
            img = img.detach().numpy()

            # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img = norm(img)
            # print(img.max(),img.min())
            img = img * 255.
            slice1_2d = img[0, 0, slice1_idx, :, :]
            slice1_2d = np.uint8(slice1_2d)
            slice1_3d = np.dstack((slice1_2d, slice1_2d, slice1_2d))
            slice2_2d = img[0, 0, slice2_idx, :, :]
            slice2_2d = np.uint8(slice2_2d)
            slice2_3d = np.dstack((slice2_2d, slice2_2d, slice2_2d))

    # generate multi-class
    mask_generator = SamAutomaticMaskGenerator(model=sam,
                                               pred_iou_thresh=0.90,
                                               # min_mask_region_area=150
                                               # points_per_side=32
                                               stability_score_thresh=0.8,
                                               )

    # masks2 generation
    masks2 = mask_generator.generate(slice2_3d)
    from script_load2d import MaskSelect, MaskFuse
    if slice1_idx > 60:
        masks2 = MaskSelect(masks2, v_min=200, v_max=20000)
    else:
        masks2 = MaskSelect(masks2, v_min=200, v_max=7000)

    if len(masks2) > 0:
        masks2.append({'segmentation': MaskFuse(masks2)})
    else:
        masks2.append({'segmentation': np.zeros_like(slice2_2d)})

    # masks1 generation
    masks1 = mask_generator.generate(slice1_3d)
    from script_load2d import MaskSelect, MaskFuse
    if slice1_idx > 60:
        masks1 = MaskSelect(masks1, v_min=200, v_max=20000)
    else:
        masks1 = MaskSelect(masks1, v_min=200, v_max=7000)

    if len(masks1) > 0:
        masks1.append({'segmentation': MaskFuse(masks1)})
    else:
        masks1.append({'segmentation': np.zeros_like(slice1_2d)})

    # mask prompt
    masks1_pr, masks2_pr = mask_prompt(masks1,masks2,slice1_3d,slice2_3d,sam)
    if len(masks1_pr) > 0:
        masks1_pr.append({'segmentation': MaskFuse(masks1_pr)})
    else:
        masks1_pr.append({'segmentation': np.zeros_like(slice1_2d)})
    if len(masks2_pr) > 0:
        masks2_pr.append({'segmentation': MaskFuse(masks2_pr)})
    else:
        masks2_pr.append({'segmentation': np.zeros_like(slice2_2d)})

    Visulize = Vis()
    Visulize._show_img(slice1_3d, masks1, ind=1)
    Visulize._show_img(slice1_3d, masks1_pr, ind=2)
    Visulize._show_img(slice2_3d, masks2, ind=3)
    Visulize._show_img(slice2_3d, masks2_pr, ind=4)
    plt.show()







def _slice_seg(test_data, sam, data1_idx = 6, data2_idx = 9, slice_idx=68):


    slice_list = slice_pair(test_data,data1_idx,data2_idx,slice_idx)
    image1, image2 = slice_list[0], slice_list[1]

    ##################### PairMask #####################################
    PairM = PairMasks(sam, image1, image2, mode='embedding')
    # PairM = PairMode(sam, image1, image2, mode='embedding')
    # masks1_cor = PairM.masks1_cor
    # masks2_cor = PairM.masks2_cor
    # import cv2
    # save_dir = r'/raid/shiqi/pair_mask samples/Cytological/sample1'
    # cv2.imwrite(save_dir+'/image_1.png',image1)
    # cv2.imwrite(save_dir+'/image_2.png',image2)
    # ind = 1
    # for m1,m2 in zip(PairM.masks1_cor,PairM.masks2_cor):
    # cv2.imwrite(save_dir+'/mask_1_{}.png'.format(str(ind)),np.uint8(m1['segmentation']))
    # cv2.imwrite(save_dir+'/mask_2_{}.png'.format(str(ind)),np.uint8(m2['segmentation']))
    # ind += 1
    Visulize = Vis()
    Visulize._show_cor_img(PairM.im1, PairM.im2, PairM.masks1_cor, PairM.masks2_cor)
    plt.show()
def gau_filter(mri_data):
    from scipy.signal import convolve2d
    from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
    from scipy.signal import gaussian

    # 将 MRI 数据的数据类型从 uint8 转换为 float32
    mri_data_float = mri_data.astype(np.float32)

    # 计算傅里叶变换
    dft = fft2(mri_data_float)
    dft_shift = fftshift(dft)

    # 定义高频数据移除的比例，取值范围 [0, 1]
    high_freq_removal_ratio = 0.5

    # 计算高斯滤波核
    rows, cols = mri_data.shape[1], mri_data.shape[2]
    crow, ccol = rows // 2, cols // 2  # 中心位置
    x = np.arange(-ccol, cols - ccol)
    y = np.arange(-crow, rows - crow)
    xx, yy = np.meshgrid(x, y)
    gaussian_filter = gaussian(cols, high_freq_removal_ratio * cols / 2.0)

    # 在频域中应用高斯滤波
    dft_shift_filtered = dft_shift * gaussian_filter[:, np.newaxis]

    # 反转傅里叶变换以回到图像域
    filtered_mri_data = np.abs(ifft2(ifftshift(dft_shift_filtered)))

    # 将滤波后的图像数据类型转换回 uint8
    filtered_mri_data = np.uint8(filtered_mri_data)
    return filtered_mri_data

def fou_filter(mri_data):
    # 进行 2D 傅里叶变换
    fft_mri_data = fft2(mri_data)

    # 设定一个高频数据移除的比例
    high_frequency_removal_ratio = 0.99 # 这个值可以根据需要进行调整

    # 计算低通滤波器半径
    radius = np.sqrt(
        (fft_mri_data.shape[1] // 2) ** 2 + (fft_mri_data.shape[2] // 2) ** 2) * high_frequency_removal_ratio

    # 创建低通滤波器
    x, y = np.meshgrid(np.arange(fft_mri_data.shape[2]), np.arange(fft_mri_data.shape[1]))
    center_x, center_y = fft_mri_data.shape[2] // 2, fft_mri_data.shape[1] // 2  # 滤波器中心
    lowpass_filter = np.exp(-((x - center_x) ** 2 + (y - center_y) ** 2) / (2 * radius ** 2))

    # 应用低通滤波器到频域数据
    filtered_data = fft_mri_data * lowpass_filter

    # 反转傅里叶变换以回到图像域
    filtered_data = np.abs(ifft2(filtered_data)).astype(np.uint8)

    # 可视化原始图像和滤波后的图像
    # plt.figure(figsize=(12, 6))
    # plt.subplot(1, 2, 1)
    # plt.imshow(mri_data[10], cmap='gray')
    # plt.title("Original MRI Image")
    #
    # plt.subplot(1, 2, 2)
    # plt.imshow(filtered_data[10], cmap='gray')
    # plt.title("Filtered MRI Image (Lowpass)")
    #
    # plt.show()

    return filtered_data

def smooth(data_3d):
    from scipy import ndimage
    # 使用连通区域标记找到不同的目标
    # 使用连通区域标记找到不同的前景区域
    data = data_3d[:]
    labeled_data, num_features = ndimage.label(data)

    # 遍历每个类别
    for label in range(1, num_features + 1):
        # 获取当前类别的连通区域信息
        region_slices = ndimage.find_objects(labeled_data == label)

        # 如果当前类别在深度方向上被切割成多个连通区域
        if len(region_slices) > 1:
            # 合并这些连通区域
            for i in range(1, len(region_slices)):
                z_start, z_end = region_slices[i - 1][0].stop, region_slices[i][0].start
                data[z_start:z_end] = label

    data = data.permute(2,0,1)
    data = data.detach().cpu().numpy()
    return data

def _load_nii(img_path, label_path):
    import nibabel as nib
    from einops import rearrange
    spatial_index = [2, 1, 0]
    num_classes = 2
    target_spacing = (2.5, 1.5, 1.5)
    img_vol = nib.load(img_path)
    img = img_vol.get_fdata().astype(np.float32).transpose(spatial_index)
    img_spacing = tuple(np.array(img_vol.header.get_zooms())[spatial_index])
    img[np.isnan(img)] = 0

    seg_vol = nib.load(label_path)
    seg = seg_vol.get_fdata().astype(np.float32).transpose(spatial_index)
    seg_spacing = tuple(np.array(seg_vol.header.get_zooms())[spatial_index])
    seg[np.isnan(seg)] = 0

    seg = rearrange(
        F.one_hot(torch.tensor(seg[:, :, :]).long(), num_classes=num_classes),
        "d h w c -> c d h w",
    ).float()  # [96, 200, 200, 2]->[2, 96, 200, 200]

    if (np.max(img_spacing) / np.min(img_spacing) > 3) or (
            np.max(target_spacing / np.min(target_spacing) > 3)
    ):
        # resize 2D
        img_tensor = F.interpolate(
            input=torch.tensor(img[:, None, :, :]),
            scale_factor=tuple([img_spacing[i] / target_spacing[i] for i in range(1, 3)]),
            mode="bicubic",
        )


        # resize 3D
        img = (
            F.interpolate(
                input=rearrange(img_tensor, f"d 1 h w -> 1 1 d h w"),
                scale_factor=(img_spacing[0] / target_spacing[0], 1, 1),
                mode="nearest",
            )
            .squeeze(0)
            .numpy()
        )

    else:
        img = (
            F.interpolate(
                input=torch.tensor(img[None, None, :, :, :]),
                scale_factor=tuple(
                    [img_spacing[i] / target_spacing[i] for i in range(3)]
                ),
                mode="trilinear",
            )
            .squeeze(0)
            .numpy()
        )  # [1,38,66,66]

    ############# transforms ####################
    intensity_range = (-2.6693046, 23.542599)
    transforms = [
        ScaleIntensityRanged(
            keys=["image"],
            a_min=intensity_range[0],
            a_max=intensity_range[1],
            b_min=0,
            b_max=1,
            clip=True,
        ),
        BinarizeLabeld(keys=["label"]),
    ]
    transforms = Compose(transforms)



    trans_dict = transforms({"image": img, "label": seg})
    img_aug, seg_aug = trans_dict["image"], trans_dict["label"]

    seg_aug = seg_aug.squeeze().argmax(0)  # [64, 160, 160]

    img_aug = img_aug.repeat(3, 1, 1, 1)  # [3, 64, 160, 160]

    return seg_aug, img_aug

def online_ft(bbox,im1, im2,predictor):
    # use im1 ROI as im2 predictor mask
    # input_box = np.array([212, 300, 350, 437])
    input_box = np.array([bbox[0],bbox[1],bbox[2]-bbox[0],bbox[3]-bbox[1]])
    predictor.set_image(im1)
    masks, scores, logits = predictor.predict(
    point_coords=None,
    point_labels=None,
    box=input_box[None, :],
    multimask_output=True,
    )
    mask_input = logits[np.argmax(scores), :, :]
    predictor.set_image(im2)
    masks, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box[None, :],
        mask_input=mask_input[None, :, :],
        multimask_output=False,
    )
    return masks


def mask_prompt(masks1, masks2, im1, im2, sam):
    predictor =  SamPredictor(sam)
    im1_ROI = []
    im2_ROI = []
    for _m in masks1[:-1]:
        if (_m['segmentation']*masks2[-1]['segmentation']).sum()/_m['area'] < 0.2:
            im1_ROI.append(_m['bbox'])
    for _m in masks2[:-1]:
        if (_m['segmentation']*masks1[-1]['segmentation']).sum()/_m['area'] < 0.2:
            im2_ROI.append(_m['bbox'])
    if len(im1_ROI) > 0:
        for _r in im1_ROI:
            mask = online_ft(_r,im1,im2,predictor)
            masks1.pop(-1)
            masks1.append({'segmentation':mask[0]})
    if len(im2_ROI) > 0:
        for _r in im2_ROI:
            mask = online_ft(_r,im2,im1,predictor)
            masks2.pop(-1)
            masks2.append({'segmentation':mask[0]})
    return masks1, masks2



if __name__ == '__main__':
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



    ###################### Load Sam Model #################################
    device = 'cuda'
    sam = sam_model_registry["vit_h"](checkpoint="/raid/shiqi/sam_pretrained/sam_vit_h_4b8939.pth")
    sam.to(device=device)

    ##################### Application #####################################
    _slice_seg(test_data,sam, data1_idx = 6, data2_idx = 9, slice_idx=9)
    # result = _nii_seg(test_data,sam,data_idx=5)
    # result = _nii_seg(test_data,sam,data_idx=7)
    # _nii_seg_multi_axis(test_data,sam)
    # _slice_prompt(test_data,sam)
    #
    ################### Load data ###################
    # image_path = "/raid/shiqi/data/Data/t2w/Patient913099765_study_0.nii.gz"
    # label_path = "/raid/shiqi/data/Data/prostate_mask/Patient913099765_study_0.nii.gz"
    # label_path = "/raid/shiqi/test_id5_a0.nii.gz"
    # seg, img = _load_nii(image_path,label_path)
    # img = img.float() # torch.Size([3, 38, 66, 66])
    # img = img[None,:1, :, :, :] # torch.Size([1, 1, 38, 66, 66])
    # img = F.interpolate(img, size=seg.shape, mode="trilinear")
    # img = img[0,0,:,:,:]
    # print(seg.shape,img.shape) # torch.Size([200, 200, 96]) torch.Size([200, 200, 96])
    #
    ################### smooth #########################
    # re = smooth(seg)
    # from script_3d import write_nii_data
    # write_nii_data(save_path="/raid/shiqi/",file_name="test_id5_smooth.nii.gz",data=re)


