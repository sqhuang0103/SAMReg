import os
import cv2
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
import argparse
import matplotlib.pyplot as plt
import nibabel as nib
from model.pair2d import PairMasks, PairMode
from model.roi_match import RoiMatching
from dataset.datasets import load_data_volume
from region_correspondence.paired_regions import PairedRegions
from region_correspondence.utils import warp_by_ddf
from model.segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
from utils import Metric, Vis, Vis_cv2, visualize_masks

def _config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", default="prostate", type=str,
        choices=["kits", "pancreas", "lits", "colon", "prostate", 'miami_prostate','FIRE', "cardiac"]
    )
    parser.add_argument(
        "--save_path",
        default="path/to/save/",
        type=str,
    )
    parser.add_argument(
        "--data_prefix",
        default="path/to/data_prefix/",
        type=str,
    )
    parser.add_argument(
        "--data_split_prefix",
        default="../datafile",
        type=str,
    )
    parser.add_argument(
        "--sam_checkpoint",
        default="path/to/snapshot/",
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
    parser.add_argument("--device", default='cuda:0', type=str)
    parser.add_argument("--sam_mode", default='vit_h', type=str)
    parser.add_argument("--fix_idx", default=6, type=int)
    parser.add_argument("--mov_idx", default=9, type=int)
    parser.add_argument("--slice_idx", default=40, type=int)
    parser.add_argument("--seg_axis", default=0, type=int)
    parser.add_argument("--num_pair", default=0, type=int)
    parser.add_argument("--multi_axis", action="store_true")
    parser.add_argument("--multi_mov", action="store_true")

    # demo.py args
    parser.add_argument("--prompt", action="store_true")
    parser.add_argument("--prompt_point", default=[80,113],type=list)
    parser.add_argument("--sam_type", default="sam_h", choices=["sam_b","sam_h", "medsam", "slimsam"], type=str)
    parser.add_argument("--jacobian_inverse", action="store_true")
    parser.add_argument("--fix_image", default='./example/intra_subject/cardiac_mr/image1_R.png', type=str)
    parser.add_argument("--mov_image", default='./example/intra_subject/cardiac_mr/image1_T.png', type=str)
    parser.add_argument("--fix_label", default='./example/intra_subject/cardiac_mr/label1_R.png', type=str)
    parser.add_argument("--mov_label", default='./example/intra_subject/cardiac_mr/label1_T.png', type=str)
    parser.add_argument("--interpolate", action="store_true")
    parser.add_argument("--ROI_type", default='pseudo_ROI', type=str, choices=['pseudo_ROI', 'label_ROI'])
    parser.add_argument("--v_min", default=200, type=float)
    parser.add_argument("--v_max", default=7000, type=float)
    parser.add_argument("--sim_criteria", default=0.80, type=float)
    parser.add_argument("--ddf_max_iter", default=int(1e4), type=int)

    args = parser.parse_args()
    return args

def get_sam_url(type):
    type = str(type)
    sam_url = {
        'sam_b': "facebook/sam-vit-base",
        'sam_l': "facebook/sam-vit-large",
        'sam_h': "facebook/sam-vit-huge",
        'medsam': "wanglab/medsam-vit-base",
        'slimsam': "nielsr/slimsam-50-uniform",
        'sam_hq': "lkeab/hq-sam/blob/main/sam_hq_vit_b.pth",
    }
    return sam_url[type]

def get_prompt_point(arr):
    return torch.tensor([arr])

def get_pair_masks(image1, image2, args, num_pair=0, mode='embedding'):
    RM = RoiMatching(image1, image2, args, url=get_sam_url(args.sam_type))
    if args.prompt:
        masks_1, masks_2 = RM.get_prompt_roi(get_prompt_point(args.prompt_point), rand_prompt=False, multi_prompt_points=False)
    else:
        RM.get_paired_roi()
        masks1 = RM.masks1
        masks2 = RM.masks2
        # PairM = PairMasks(image1, image2, args, mode='embedding')
        # masks1 = PairM.masks1_cor
        # masks2 = PairM.masks2_cor
        if len(masks1) < num_pair:
            num_pair = 0 #default
        if num_pair == 0:
            masks_1 = masks1[:len(masks1) // 2 + 1]
            masks_2 = masks2[:len(masks2) // 2 + 1]
        else:
            masks_1 = masks1[:num_pair]
            masks_2 = masks2[:num_pair]
    return masks_1, masks_2

def get_image(path):
    img = cv2.imread(path)
    img = (img - img.min()) / (img.max() - img.min())
    img = np.uint8(img * 255.)
    return img

def get_PIL_image(path):
    img = Image.open(path).convert("RGB")
    img = img.resize((200, 200))
    return img

def get_label(path):
    img = cv2.imread(path, 0)
    # img[img < 255] = 1
    # img[img == 255] = 0
    img = cv2.resize(img, (200,200), interpolation=cv2.INTER_AREA)
    return img.astype('bool')


def get_ddf(masks1, masks2, args):
    def list_to_tensor(items):
        tensors = [torch.tensor(item, dtype=torch.float) for item in items]
        stacked_tensor = torch.stack(tensors, dim=0)
        return stacked_tensor

    masks_mov = list_to_tensor(masks2)
    masks_fix = list_to_tensor(masks1)
    paired_rois = PairedRegions(masks_mov=masks_mov, masks_fix=masks_fix, device=args.device)
    ddf = paired_rois.get_dense_correspondence(transform_type='ddf', max_iter=args.ddf_max_iter, lr=1e-3, w_ddf=1.0,
                                               verbose=True)
    return ddf

def warp(mask,ddf, args):
    mask = torch.tensor(mask).unsqueeze(0)
    masks_warped = (warp_by_ddf(mask.to(dtype=torch.float32, device=args.device), ddf) * 255).to(
        torch.uint8)  # torch.Size([1, 200, 200])
    masks_warped = masks_warped.cpu().numpy()
    return masks_warped.astype(bool)

class Resize():
    def __init__(self,img):
        self.img = img
        self.img = np.array(self.img, dtype=np.float32)
        self.H,self.W = img.shape[0], img.shape[1]
    def zoom_out(self,in_img,beta=2):
        in_img = np.array(in_img, dtype=np.float32)
        new_height = self.H // beta
        new_width = self.W // beta
        out_img = cv2.resize(in_img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return out_img

    def recovery(self,out_img):
        print(out_img.shape, self.W, self.H)
        out_img = np.array(out_img, dtype=np.float32)
        in_img = cv2.resize(out_img, (self.W, self.H), interpolation=cv2.INTER_CUBIC)
        in_img = np.uint8(in_img>0)*255.
        return in_img





if __name__ == '__main__':
    args = _config()
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    metric = Metric()
    fix_image = get_PIL_image(args.fix_image)
    mov_image = get_PIL_image(args.mov_image)

    fix_masks, mov_masks = get_pair_masks(fix_image, mov_image, args=args, num_pair=args.num_pair)
    fix_image = np.array(fix_image).astype(np.uint8)
    mov_image = np.array(mov_image).astype(np.uint8)
    visualized_image1, visualized_image2 = visualize_masks(fix_image, fix_masks, mov_image, mov_masks)
    # print(fix_image.shape)
    cv2.imwrite('/home/shiqi/SAMReg/fixed.png', visualized_image1)
    cv2.imwrite('/home/shiqi/SAMReg/moving.png', visualized_image2)
    # visualization = Vis()
    visualization = Vis_cv2()
    # visualization._show_cor_img(fix_image, mov_image, fix_masks, mov_masks)


    if args.interpolate:
        match args.ROI_type:
            case 'pseudo_ROI':
                idx = np.random.randint(0, len(fix_masks))
                mov_label = mov_masks[idx]
                fix_label = fix_masks[idx]
                cv2.imwrite('/home/shiqi/PromptReg/fix_label_1.png',np.uint8(fix_label)*255.)
                cv2.imwrite('/home/shiqi/PromptReg/mov_label_1.png',np.uint8(mov_label)*255.)
            case 'label_ROI':
                fix_label = get_label(args.fix_label)
                mov_label = get_label(args.mov_label)
                cv2.imwrite('/home/shiqi/PromptReg/fix_label.png', np.uint8(fix_label) * 255.)
                cv2.imwrite('/home/shiqi/PromptReg/mov_label.png', np.uint8(mov_label) * 255.)

        # R = Resize(mov_label)
        # fix_masks = [R.zoom_out(item) for item in fix_masks]
        # mov_masks = [R.zoom_out(item) for item in mov_masks]
        # mov_label = R.zoom_out(mov_label)
        # fix_label = R.zoom_out(fix_label)

        ######## delete
        moved_list = ['norm','semi','fully']
        for _m in moved_list:
            if _m == 'norm':
                ddf = get_ddf(fix_masks, mov_masks, args)
                wraped_seg = warp(mov_label, ddf, args)  # (1,200,200)
                metric.update(wraped_seg[0], fix_label)
                cv2.imwrite('/home/shiqi/PromptReg/moved_norm_1.png', np.uint8(wraped_seg[0]) * 255.)
            elif _m == 'semi':
                fix_masks.append(fix_label)
                mov_masks.append(mov_label)
                ddf = get_ddf(fix_masks, mov_masks, args)
                wraped_seg = warp(mov_label, ddf, args)  # (1,200,200)
                cv2.imwrite('/home/shiqi/PromptReg/moved_semi_1.png', np.uint8(wraped_seg[0]) * 255.)
            elif _m == 'fully':
                ddf = get_ddf([fix_label], [mov_label], args)
                wraped_seg = warp(mov_label, ddf, args)  # (1,200,200)
                cv2.imwrite('/home/shiqi/PromptReg/moved_fully_1.png', np.uint8(wraped_seg[0]) * 255.)


        # ddf = get_ddf(fix_masks, mov_masks, args)
        # wraped_seg = warp(mov_label, ddf, args)  # (1,200,200)
        # metric.update(wraped_seg[0], fix_label)

        # w = wraped_seg[0]
        # w = R.recovery(w)
        # cv2.imwrite('/home/shiqi/PromptReg/moved.png',np.uint8(wraped_seg[0])*255.)

        visualization._show_interpolate_img(mov_label,wraped_seg[0],fix_label)
        print('Dice: {:.4f}; TRE: {:.4f}'.format(metric.get_dice()[0], metric.get_tre()[0]))

    # plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()








