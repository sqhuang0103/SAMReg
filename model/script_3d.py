import os.path

import numpy as np

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
# from modeling.Med_SAM.image_encoder import ImageEncoderViT_3d_v2 as ImageEncoderViT_3d
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import matplotlib.pyplot as plt

import monai
# from monai.transforms import Compose, LoadImaged, Spacingd, ToTensord, Resized, AddChanneld
# from monai.data import write_nifti
import nibabel as nib
from scipy import ndimage
# import scipy.ndimage as ndimage
import scipy.ndimage.morphology as morphology




def sam3d():

        sam = sam_model_registry["vit_b"](checkpoint="/raid/shiqi/sam_pretrained/sam_vit_b_01ec64.pth")
        mask_generator = SamAutomaticMaskGenerator(sam)

        img_encoder = ImageEncoderViT_3d(
                depth=12,
                embed_dim=768,
                img_size=1024,
                mlp_ratio=4,
                norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
                num_heads=12,
                patch_size=16,
                qkv_bias=True,
                use_rel_pos=True,
                global_attn_indexes=[2, 5, 8, 11],
                window_size=14,
                cubic_window_size=8,
                out_chans=256,
                num_slice = 16)

        img_encoder.load_state_dict(mask_generator.predictor.model.image_encoder.state_dict(), strict=False)
        del sam

def load_nii_data(path):


        # 定义要应用的转换
        # LoadImaged 读取 nii.gz 文件
        # AddChanneld 在图像上添加一个通道维度，这在训练神经网络时常常是必需的
        # ToTensord 将图像转换为 PyTorch 张量
        transforms = Compose([
                LoadImaged(keys=["image"]),  # 加载图像
                AddChanneld(keys=["image"]),  # 增加通道维度
                Resized(keys=["image"], spatial_size=(96, 200, 200)),  # 调整图像大小
                ToTensord(keys=["image"])  # 将图像转换为张量
        ])
        # 应用转换
        # 注意：MONAI 期望输入是一个包含图像文件名的字典
        dataset = monai.data.Dataset(data=[{"image": path}], transform=transforms)
        sample = dataset[0]
        image_tensor = sample["image"]

        # 输出张量信息
        # print(tensor_image.shape)  # 查看张量的形状 # (1,96,200,200)
        # print(tensor_image.min(),tensor_image.max())  # 查看张量的最值 #tensor(-1.4998) tensor(4.6097)
        # print(tensor_image.dtype)  # 查看张量的数据类型 # torch.float32

        return image_tensor

def write_nii_data(save_path,file_name,data):
    # data: nd.array
    # 将 NumPy 数组转换为 NIfTI 图像对象
    nifti_image = nib.Nifti1Image(data, np.eye(4))

    # 保存为 nii.gz 文件
    save_path = os.path.join(save_path,file_name)
    nib.save(nifti_image, save_path)
    print(f"Saved the NIfTI image to {save_path}")

def dataset_nii(file_path):
        # 定义图像变换
        transforms = Compose([
                LoadImaged(keys=["image"]),  # 加载 nii.gz 文件
                Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode='bilinear'),  # 调整图像间距
                ToTensord(keys=["image"])  # 转换为 PyTorch 张量
        ])


        # 使用 MONAI 的 Dataset 来处理数据
        data = [{"image": file_path}]
        dataset = monai.data.Dataset(data, transform=transforms)

        # 从 Dataset 中获取数据
        data_dict = dataset[0]
        image_tensor = data_dict["image"]

        # 调整张量形状，从 (200, 200, 96) 转换为 (96, 200, 200)
        image_tensor = image_tensor.permute(2, 0, 1)

        return image_tensor

def load_sam():
        sam = sam_model_registry["vit_h"](checkpoint="/raid/shiqi/sam_pretrained/sam_vit_h_4b8939.pth")
        sam.to(device='cuda')
        mask_generator = SamAutomaticMaskGenerator(model=sam,
                                                   pred_iou_thresh=0.90,
                                                   # min_mask_region_area=150
                                                   # points_per_side=32
                                                   stability_score_thresh=0.9,
                                                   )
        return mask_generator

from .script_load2d import MaskSelect, MaskFuse

def sam_slice(net,data,axis=0):

        # data shape (D,H,W)
        result_3d = np.empty_like(data)
        for i in range(data.shape[axis]):
                # 获取单个切片
                if axis == 0:
                    slice_2d = data[i]
                elif axis == 1:
                    slice_2d = data[:,i,:]
                elif axis == 2:
                    slice_2d = data[:, :, i]

                # slice_2d = (slice_2d-slice_2d.min())/(slice_2d.max()-slice_2d.min()+1e-5)
                # slice_2d *= 255.
                slice_2d = np.uint8(slice_2d)
                # print(slice_2d.shape)

                # 将切片扩展为 3 通道，形状为 (200, 200,3)
                slice_3_channel = np.dstack((slice_2d, slice_2d, slice_2d))
                # slice_3_channel = slice_3_channel.permute(1,2,0)
                # print(slice_3_channel.shape)

                # 输入到 2D 网络
                masks = net.generate(slice_3_channel)
                if i > 60:
                    masks = MaskSelect(masks,v_min=200, v_max=20000)
                else:
                    masks = MaskSelect(masks,v_min=200, v_max=7000)

                if len(masks) > 0:
                    masks.append({'segmentation': MaskFuse(masks)})
                else:
                    masks.append({'segmentation': np.zeros_like(slice_2d)})

                # output_2d = np.uint8(masks[-1]['segmentation']>0)
                output_2d = np.uint8(masks[-1]['segmentation'])
                # print(output_2d.shape)

                # 移除批次维度并存储输出
                if axis == 0:
                    result_3d[i] = output_2d
                elif axis == 1:
                    result_3d[:,i,:] = output_2d
                elif axis == 2:
                    result_3d[:,:,i] = output_2d
        return result_3d


def sam_slice_v2(net, data, axis=0):
    # data shape (D,H,W)
    previous_im, previous_mask = None, None
    result_3d = np.empty_like(data)
    for i in range(data.shape[axis]):
        # 获取单个切片
        if axis == 0:
            slice_2d = data[i]
        elif axis == 1:
            slice_2d = data[:, i, :]
        elif axis == 2:
            slice_2d = data[:, :, i]

        # slice_2d = (slice_2d-slice_2d.min())/(slice_2d.max()-slice_2d.min()+1e-5)
        # slice_2d *= 255.
        slice_2d = np.uint8(slice_2d)
        # print(slice_2d.shape)

        # 将切片扩展为 3 通道，形状为 (200, 200,3)
        slice_3_channel = np.dstack((slice_2d, slice_2d, slice_2d))
        # slice_3_channel = slice_3_channel.permute(1,2,0)
        # print(slice_3_channel.shape)

        # 输入到 2D 网络
        masks = net.generate(slice_3_channel)
        if i > 60:
            masks = MaskSelect(masks, v_min=200, v_max=20000)
        else:
            masks = MaskSelect(masks, v_min=200, v_max=7000)

        if len(masks) > 0:
            masks.append({'segmentation': MaskFuse(masks)})
        else:
            masks.append({'segmentation': np.zeros_like(slice_2d)})

        from script_3d_slice import mask_prompt
        if previous_mask is not None:
            masks, previous_mask = mask_prompt(masks, previous_mask,slice_3_channel, previous_im)
            if len(masks) > 0:
                masks.append({'segmentation': MaskFuse(masks)})
            else:
                masks.append({'segmentation': np.zeros_like(slice_2d)})
            if len(previous_mask) > 0:
                previous_mask.append({'segmentation': MaskFuse(previous_mask)})
            else:
                previous_mask.append({'segmentation': np.zeros_like(slice_2d)})

            # output_2d = np.uint8(masks[-1]['segmentation']>0)
        output_2d = np.uint8(masks[-1]['segmentation'])
        previous_output_2d = np.uint8(previous_mask[-1]['segmentation'])
        previous_mask = masks
        previous_im = slice_3_channel
        # print(output_2d.shape)

        # 移除批次维度并存储输出
        if axis == 0:
            result_3d[i] = output_2d
            result_3d[i-1] = previous_output_2d
        elif axis == 1:
            result_3d[:, i, :] = output_2d
            result_3d[:, i-1, :] = previous_output_2d
        elif axis == 2:
            result_3d[:, :, i] = output_2d
            result_3d[:, :, i-1] = previous_output_2d
    return result_3d

def multi_axis_slice(img_path,save_path):
    net = load_sam()
    img = load_nii_data(img_path)
    img = img.detach().numpy()
    img = img[0]
    # print('load data:',img.shape) #(1,96,200,200)
    result_3d_a0 = sam_slice(net, img, axis=0)
    write_nii_data(save_path, 'test_a0.nii.gz', result_3d_a0)
    result_3d_a1 = sam_slice(net, img, axis=1)
    write_nii_data(save_path, 'test_a1.nii.gz', result_3d_a1)
    result_3d_a2 = sam_slice(net, img, axis=2)
    write_nii_data(save_path, 'test_a2.nii.gz', result_3d_a2)

    result_3d = np.zeros_like(result_3d_a0)

    result_3d += result_3d_a0
    result_3d += result_3d_a1
    result_3d ++ result_3d_a2

    result_3d = np.uint8(result_3d > 0)
    write_nii_data(save_path, 'test_3d.nii.gz', result_3d)

    # print(result_3d.min(),result_3d.max())
    _show_image(img, result_3d, 10, 1)
    _show_image(img, result_3d, 45, 2)
    _show_image(img, result_3d, 80, 3)
    plt.show()

def smooth_v1(output_data):
    min_overlap_ratio = 0.8

    # 遍历每个切片
    for z in range(output_data.shape[0]):
        # 使用连通区域标记找到不同的前景区域
        labeled_data, num_features = ndimage.label(output_data[z])

        # 对于每对前景区域，检查它们之间的重叠并合并
        for label1 in range(1, num_features + 1):
            region1 = (labeled_data == label1)
            for label2 in range(label1 + 1, num_features + 1):
                region2 = (labeled_data == label2)

                # 计算重叠区域的像素数
                overlap_pixels = np.sum(np.logical_and(region1, region2))

                # 计算重叠的比例
                overlap_ratio = overlap_pixels / np.sum(region1)

                # 如果重叠比例大于等于阈值，则合并两个区域
                if overlap_ratio >= min_overlap_ratio:
                    labeled_data[np.logical_or(region1, region2)] = label1

        # 更新切片上的输出数据
        output_data[z] = labeled_data
    output_data = np.uint8(output_data>0)

    return output_data

def smooth_v2(output_data):
    # 定义连接同一类别的目标的深度阈值（可以根据需要进行调整）
    depth_threshold = 5  # 如果同一类别的目标在深度方向上分隔不超过5个切片，则认为它们是同一目标

    # 使用连通区域标记找到不同的目标
    labeled_data, num_features = ndimage.label(output_data)

    # 遍历每个类别
    for label in range(1, num_features + 1):
        label_mask = (labeled_data == label)

        # 找到目标在深度方向上的切片范围
        slices = ndimage.find_objects(label_mask)
        z_start, z_end = slices[0][0].start, slices[-1][0].stop

        # 如果深度范围小于阈值，则认为是同一目标
        if z_end - z_start <= depth_threshold:
            continue

        # 合并同一类别的目标
        for z in range(z_start, z_end):
            # 找到同一类别的目标在当前深度切片上的区域
            region = label_mask[z]

            # 使用逻辑或将目标合并到第一个深度切片上
            labeled_data[z_start] |= region

    # 更新分割结果数据
    return labeled_data

def _smooth_3d(save_path,source_name='test_a0.nii.gz',save_name='test_a0_smooth.nii.gz'):
    data = load_nii_data(os.path.join(save_path, source_name))
    data = data.detach().numpy()
    data = data[0]

    output = smooth_v2(data)
    write_nii_data(save_path, save_name, output)

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.4])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def _show_image(image,mask,ind,fig_ind):
        plt.figure(fig_ind)
        plt.imshow(image[ind])
        show_mask(mask[ind], plt.gca(), random_color=False)
        plt.axis('off')




if __name__ == '__main__':
    img_path = r"/raid/shiqi/data/Data/t2w/Patient001061633_study_0.nii.gz"
    seg_path = r"/raid/shiqi/data/Data/prostate_mask/Patient001061633_study_0.nii.gz"
    save_path = r"/raid/shiqi"

    multi_axis_slice(img_path,save_path)
    # _smooth_3d(save_path,
    #           source_name='test_a1.nii.gz',save_name='test_a1_smooth.nii.gz'
    #  )






