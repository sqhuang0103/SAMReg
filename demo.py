import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F
import argparse
import matplotlib.pyplot as plt
import nibabel as nib
from model.pair2d import PairMasks, PairMode
from dataset.datasets import load_data_volume
from region_correspondence.paired_regions import PairedRegions
from region_correspondence.utils import warp_by_ddf
from model.segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
from utils import Metric, Vis

def _config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", default="prostate", type=str,
        choices=["kits", "pancreas", "lits", "colon", "prostate", 'miami_prostate','FIRE', "cardiac"]
    )
    parser.add_argument(
        "--save_path",
        default="/raid/shiqi/RegProstate",
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
        "--sam_checkpoint",
        default="/raid/shiqi/sam_pretrained/sam_vit_h_4b8939.pth",
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
        default="prostate_mask",
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
    parser.add_argument("--data_dim", default=2, type=int)
    parser.add_argument("--device", default='cuda', type=str)
    parser.add_argument("--sam_mode", default='vit_h', type=str)
    parser.add_argument("--fix_idx", default=6, type=int)
    parser.add_argument("--mov_idx", default=9, type=int)
    parser.add_argument("--slice_idx", default=40, type=int)
    parser.add_argument("--seg_axis", default=0, type=int)
    parser.add_argument("--num_pair", default=0, type=int)
    parser.add_argument("--multi_axis", action="store_true")
    parser.add_argument("--multi_mov", action="store_true")

    # demo args
    parser.add_argument("--fix_image", default='./example/cardiac_2d/image1.png', type=str)
    parser.add_argument("--mov_image", default='./example/cardiac_2d/image2.png', type=str)
    parser.add_argument("--fix_label", default='./example/cardiac_2d/label1.png', type=str)
    parser.add_argument("--mov_label", default='./example/cardiac_2d/label2.png', type=str)
    parser.add_argument("--interpolate", action="store_true")
    parser.add_argument("--ROI_type", default='pseudo_ROI', type=str, choices=['pseudo_ROI', 'label_ROI'])




    args = parser.parse_args()
    return args

def get_pair_masks(sam, image1, image2, num_pair=0, mode='embedding'):
    # TODO: randpoint mode
    PairM = PairMasks(sam, image1, image2, mode='embedding')
    if len(PairM.masks1_cor) < num_pair:
        num_pair = 0 #default
    if num_pair == 0:
        masks_1 = PairM.masks1_cor[:len(PairM.masks1_cor) // 2 + 1]
        masks_2 = PairM.masks2_cor[:len(PairM.masks2_cor) // 2 + 1]
    else:
        masks_1 = PairM.masks1_cor[:num_pair]
        masks_2 = PairM.masks2_cor[:num_pair]

    return masks_1, masks_2

def get_image(path):
    img = cv2.imread(path)
    img = (img - img.min()) / (img.max() - img.min())
    img = np.uint8(img * 255.)
    return img

def get_label(path):
    img = cv2.imread(path, 0)
    return img.astype('bool')


def get_ddf(masks1, masks2, args):
    def list_to_tensor(items):
        # 将列表中的每个NumPy数组转换为PyTorch张量
        tensors = [torch.tensor(item['segmentation'], dtype=torch.float) for item in items]
        # 沿着新维度堆叠张量，得到形状为(C, H, W)的张量
        stacked_tensor = torch.stack(tensors, dim=0)
        return stacked_tensor

    masks_mov = list_to_tensor(masks2)
    masks_fix = list_to_tensor(masks1)
    paired_rois = PairedRegions(masks_mov=masks_mov, masks_fix=masks_fix, device=args.device)
    ddf = paired_rois.get_dense_correspondence(transform_type='ddf', max_iter=int(1e4), lr=1e-3, w_ddf=1.0,
                                               verbose=True)
    return ddf

def warp(mask,ddf, args):
    mask = torch.tensor(mask).unsqueeze(0)
    masks_warped = (warp_by_ddf(mask.to(dtype=torch.float32, device=args.device), ddf) * 255).to(
        torch.uint8)  # torch.Size([1, 200, 200])
    masks_warped = masks_warped.cpu().numpy()
    return masks_warped.astype(bool)

def recover_3d(mov_seg,moved_slice, axis, slice_idx):
    '''mov_seg: (D,H,W)'''
    match axis:
        case 0:
            mov_seg[slice_idx] += moved_slice
        case 1:
            mov_seg[:,slice_idx,:] += moved_slice
        case 2:
            mov_seg[:,:,slice_idx] += moved_slice
    return mov_seg



if __name__ == '__main__':
    args = _config()
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    metric = Metric()
    sam = sam_model_registry[args.sam_mode](checkpoint=args.sam_checkpoint)
    sam.to(device=args.device)

    fix_image = get_image(args.fix_image)
    mov_image = get_image(args.mov_image)
    fix_masks, mov_masks = get_pair_masks(sam, fix_image, mov_image)
    print(len(fix_masks),len(mov_masks))
    visualization = Vis()
    visualization._show_cor_img(fix_image,mov_image,fix_masks,mov_masks)


    if True:
    # if args.interpolate:
        match args.ROI_type:
            case 'pseudo_ROI':
                idx = np.random.randint(0, len(fix_masks))
                mov_label = mov_masks[idx]['segmentation']
                fix_label = fix_masks[idx]['segmentation']
            case 'label_ROI':
                fix_label = get_label(args.fix_label)
                mov_label = get_label(args.mov_label)

        ddf = get_ddf(fix_masks, mov_masks, args)
        wraped_seg = warp(mov_label, ddf, args)  # (1,200,200)
        metric.update(wraped_seg, fix_label)
        print('Dice: {:.4f}; TRE: {:.4f}'.format(metric.get_dice()[0], metric.get_tre()[0]))

    # plt.show()








