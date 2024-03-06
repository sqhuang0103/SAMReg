import pickle
import os, sys

import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
import nibabel as nib
import torch.nn.functional as F
from .base_dataset import BaseVolumeDataset
from einops import rearrange
import pandas


class KiTSVolumeDataset(BaseVolumeDataset):
    def _set_dataset_stat(self):
        self.intensity_range = (-54, 247)
        self.target_spacing = (2, 1.5, 1.5)
        self.global_mean = 59.53867
        self.global_std = 55.457336
        self.spatial_index = [0, 1, 2]  # index used to convert to DHW
        self.do_dummy_2D = False
        self.target_class = 2


class LiTSVolumeDataset(BaseVolumeDataset):
    def _set_dataset_stat(self):
        self.intensity_range = (-48, 163)
        self.target_spacing = (2.5, 1.5, 1.5)
        self.global_mean = 60.057533
        self.global_std = 40.198017
        self.spatial_index = [2, 1, 0]  # index used to convert to DHW
        self.do_dummy_2D = False
        self.target_class = 2


class PancreasVolumeDataset(BaseVolumeDataset):
    def _set_dataset_stat(self):
        self.intensity_range = (-39, 204)
        self.target_spacing = (2.5, 1.5, 1.5)
        self.global_mean = 68.45214
        self.global_std = 63.422806
        self.spatial_index = [2, 1, 0]  # index used to convert to DHW
        self.do_dummy_2D = True
        self.target_class = 2


class ColonVolumeDataset(BaseVolumeDataset):
    def _set_dataset_stat(self):
        self.intensity_range = (-57, 175)
        self.target_spacing = (2.5, 1.5, 1.5)
        self.global_mean = 65.175035
        self.global_std = 32.651197
        self.spatial_index = [2, 1, 0]  # index used to convert to DHW
        self.do_dummy_2D = True
        self.target_class = 1

class ProstateVolumeDataset(BaseVolumeDataset):
    def _set_dataset_stat(self):
        self.intensity_range = (-2.6693046, 23.542599)
        self.target_spacing = (2.5, 1.5, 1.5)
        self.global_mean = 0.
        self.global_std = 0.
        self.spatial_index = [2, 1, 0]  # index used to convert to DHW
        self.do_dummy_2D = True
        self.target_class = 1
        # self.num_classes = 3
class MiamiProstateVolumeDataset(BaseVolumeDataset):
    def _set_dataset_stat(self):
        self.intensity_range = (0.0, 23662.0)
        self.target_spacing = (2.5, 1.5, 1.5)
        self.global_mean = 438.26965
        self.global_std = 339.63736
        self.spatial_index = [2, 1, 0]  # index used to convert to DHW
        self.do_dummy_2D = True
        self.target_class = 1

class BrainVolumeDataset(BaseVolumeDataset):
    def _set_dataset_stat(self):
        self.intensity_range = (-54, 247)
        self.target_spacing = (2, 1.5, 1.5)
        self.global_mean = 59.53867
        self.global_std = 55.457336
        self.spatial_index = [0, 1, 2]  # index used to convert to DHW
        self.do_dummy_2D = False
        self.target_class = 2

class CardiacVolumeDataset(BaseVolumeDataset):
    def __init__(
        self,
        image_paths,
        label_meta,
        augmentation,
        batch_size,
        split="train",
        rand_crop_spatial_size=(96, 96, 96),
        convert_to_sam=True,
        do_test_crop=True,
        do_nnunet_intensity_aug=True,
        multi_class=False,
    ):
        super().__init__(
            image_paths,
            label_meta,
            augmentation,
            batch_size,
            split=split,
            rand_crop_spatial_size=rand_crop_spatial_size,
            convert_to_sam=convert_to_sam,
            do_test_crop=do_test_crop,
            do_nnunet_intensity_aug=do_nnunet_intensity_aug,)
        self.img_dict = image_paths['fix']
        self.img_mov_dict = image_paths['mov']
        self.label_dict = label_meta['fix']
        self.label_mov_dict = label_meta['mov']


    def _set_dataset_stat(self):
        self.intensity_range = (-57, 175)
        self.target_spacing = (2.5, 1.5, 1.5)
        self.global_mean = 65.175035
        self.global_std = 32.651197
        self.spatial_index = [2, 1, 0]  # index used to convert to DHW
        self.do_dummy_2D = True
        self.target_class = 1
        # TODO: only for train segmentation model
        # self.target_class = None
        # self.num_classes = 4

    def __getitem__(self, idx):
        img_path = self.img_dict[idx]
        img_mov_path = self.img_mov_dict[idx]
        label_path = self.label_dict[idx]
        label_mov_path = self.label_mov_dict[idx]

        img_vol = nib.load(img_path)
        img = img_vol.get_fdata().astype(np.float32).transpose(self.spatial_index)
        img_spacing = tuple(np.array(img_vol.header.get_zooms())[self.spatial_index])

        img_mov_vol = nib.load(img_mov_path)
        img_mov = img_mov_vol.get_fdata().astype(np.float32).transpose(self.spatial_index)

        seg_vol = nib.load(label_path)
        seg = seg_vol.get_fdata().astype(np.float32).transpose(self.spatial_index)
        seg_spacing = tuple(np.array(seg_vol.header.get_zooms())[self.spatial_index])

        seg_mov_vol = nib.load(label_mov_path)
        seg_mov = seg_mov_vol.get_fdata().astype(np.float32).transpose(self.spatial_index)

        img[np.isnan(img)] = 0 #(96,200,200)
        seg[np.isnan(seg)] = 0 #(96,200,200)
        img_mov[np.isnan(img_mov)] = 0 #(96,200,200)
        seg_mov[np.isnan(seg_mov)] = 0 #(96,200,200)
        # print('seg: ', seg.max(), seg_mov.max())
        # print('img: ', img.shape, img_mov.shape)


        if self.target_class is not None:
            seg = (seg == self.target_class).astype(np.float32)
            seg_mov = (seg_mov == self.target_class).astype(np.float32)
            self.num_classes = 2

        assert self.num_classes is not None

        seg = rearrange(
            F.one_hot(torch.tensor(seg[:, :, :]).long(), num_classes=self.num_classes),
            "d h w c -> c d h w",
        ).float() #[96, 200, 200, 2]->[2, 96, 200, 200]

        seg_mov = rearrange(
            F.one_hot(torch.tensor(seg_mov[:, :, :]).long(), num_classes=self.num_classes),
            "d h w c -> c d h w",
        ).float()  # [96, 200, 200, 2]->[2, 96, 200, 200]
        # print('seg: ', seg.shape, seg_mov.shape)


        if (np.max(img_spacing) / np.min(img_spacing) > 3) or (
            np.max(self.target_spacing / np.min(self.target_spacing) > 3)
        ):
            # resize 2D
            img_tensor = F.interpolate(
                input=torch.tensor(img[:, None, :, :]),
                scale_factor=tuple([img_spacing[i] / self.target_spacing[i] for i in range(1, 3)]),
                mode="bicubic",
            )

            img_mov_tensor = F.interpolate(
                input=torch.tensor(img_mov[:, None, :, :]),
                scale_factor=tuple([img_spacing[i] / self.target_spacing[i] for i in range(1, 3)]),
                mode="bicubic",
            )

            if self.split != "test":
                seg_tensor = F.interpolate(
                    input=rearrange(seg, "c d h w -> d c h w"),
                    scale_factor=tuple(
                        [img_spacing[i] / self.target_spacing[i] for i in range(1, 3)]
                    ),
                    mode="bilinear",
                )

                seg_mov_tensor = F.interpolate(
                    input=rearrange(seg_mov, "c d h w -> d c h w"),
                    scale_factor=tuple(
                        [img_spacing[i] / self.target_spacing[i] for i in range(1, 3)]
                    ),
                    mode="bilinear",
                )

            # resize 3D
            img = (
                F.interpolate(
                    input=rearrange(img_tensor, f"d 1 h w -> 1 1 d h w"),
                    scale_factor=(img_spacing[0] / self.target_spacing[0], 1, 1),
                    mode="nearest",
                )
                .squeeze(0)
                .numpy()
            )

            img_mov = (
                F.interpolate(
                    input=rearrange(img_mov_tensor, f"d 1 h w -> 1 1 d h w"),
                    scale_factor=(img_spacing[0] / self.target_spacing[0], 1, 1),
                    mode="nearest",
                )
                .squeeze(0)
                .numpy()
            )

            if self.split != "test":
                seg = (
                    F.interpolate(
                        input=rearrange(seg_tensor, f"d c h w -> 1 c d h w"),
                        scale_factor=(img_spacing[0] / self.target_spacing[0], 1, 1),
                        mode="nearest",
                    )
                    .squeeze(0)
                    .numpy()
                )

                seg_mov = (
                    F.interpolate(
                        input=rearrange(seg_mov_tensor, f"d c h w -> 1 c d h w"),
                        scale_factor=(img_spacing[0] / self.target_spacing[0], 1, 1),
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
                        [img_spacing[i] / self.target_spacing[i] for i in range(3)]
                    ),
                    mode="trilinear",
                )
                .squeeze(0)
                .numpy()
            ) # [1,38,66,66]

            img_mov = (
                F.interpolate(
                    input=torch.tensor(img_mov[None, None, :, :, :]),
                    scale_factor=tuple(
                        [img_spacing[i] / self.target_spacing[i] for i in range(3)]
                    ),
                    mode="trilinear",
                )
                .squeeze(0)
                .numpy()
            )  # [1,38,66,66]

            if self.split != "test":
                seg = (
                    F.interpolate(
                        input=seg.unsqueeze(0),
                        scale_factor=tuple(
                            [img_spacing[i] / self.target_spacing[i] for i in range(3)]
                        ),
                        mode="trilinear",
                    )
                    .squeeze(0)
                    .numpy()
                ) # [1,38,66,66]

                seg_mov = (
                    F.interpolate(
                        input=seg_mov.unsqueeze(0),
                        scale_factor=tuple(
                            [img_spacing[i] / self.target_spacing[i] for i in range(3)]
                        ),
                        mode="trilinear",
                    )
                    .squeeze(0)
                    .numpy()
                )  # [1,38,66,66]


        if self.aug and self.split == "train":
            trans_dict = self.transforms({"image": img, "label": seg})[0]
            trans_mov_dict = self.transforms({"image": img_mov, "label": seg_mov})[0]
            img_aug, seg_aug = trans_dict["image"], trans_dict["label"]
            img_mov_aug, seg_mov_aug = trans_mov_dict["image"], trans_mov_dict["label"]

        else:
            trans_dict = self.transforms({"image": img, "label": seg})
            trans_mov_dict = self.transforms({"image": img_mov, "label": seg_mov})
            img_aug, seg_aug = trans_dict["image"], trans_dict["label"]
            img_mov_aug, seg_mov_aug = trans_mov_dict["image"], trans_mov_dict["label"]
        # print('seg: ', seg_aug.shape, seg_mov_aug.shape)

        seg_aug = seg_aug.squeeze().argmax(0) #[64, 160, 160]
        seg_mov_aug = seg_mov_aug.squeeze().argmax(0) #[64, 160, 160]

        img_aug = img_aug.repeat(3, 1, 1, 1)  #[3, 64, 160, 160]
        img_mov_aug = img_mov_aug.repeat(3, 1, 1, 1)  #[3, 64, 160, 160]

        if self.convert_to_sam:
            pass

        return [img_aug,img_mov_aug], [seg_aug,seg_mov_aug], torch.tensor(img_spacing)


DATASET_DICT = {
    "kits": KiTSVolumeDataset,
    "lits": LiTSVolumeDataset,
    "msd": PancreasVolumeDataset,
    "colon": ColonVolumeDataset,
    "prostate": ProstateVolumeDataset,
    "miami_prostate": ProstateVolumeDataset,
    "brain": BrainVolumeDataset,
    "cardiac":CardiacVolumeDataset,
}


def load_data_volume(
    *,
    data,
    task,
    path_prefix,
    split_prefix,
    batch_size,
    data_dir=None,
    split="train",
    deterministic=False,
    augmentation=False,
    fold=0,
    rand_crop_spatial_size=(96, 96, 96),
    convert_to_sam=False,
    do_test_crop=True,
    do_nnunet_intensity_aug=False,
    num_worker=4,
    multi_class=False,
):
    if not path_prefix:
        raise ValueError("unspecified data directory")
    if data_dir is None:
        data_dir = os.path.join(split_prefix,data,split+'.csv')

    # with open(data_dir, "rb") as f:
    #     d = pickle.load(f)[fold][split]
    df = pd.read_csv(data_dir)

    match data:
        case 'prostate':
            img_files = [os.path.join(path_prefix, p) for p in df['t2w']]
            seg_files = [os.path.join(path_prefix, p) for p in df[task]]
            if multi_class:
                seg_files_1 = [os.path.join(path_prefix, p) for p in df["prostate_mask"]]
                seg_files_2 = [os.path.join(path_prefix, p) for p in df["zonal_mask"]]
                seg_files_3 = [os.path.join(path_prefix, p) for p in df["lesion"]]
                seg_files = [seg_files_1,seg_files_2,seg_files_3]

        case 'miami_prostate':
            img_files = [os.path.join(path_prefix, p) for p in df['t2-0']]
            seg_files = [os.path.join(path_prefix, p) for p in df[task]]
        case 'cardiac':
            img_files = {'fix':[os.path.join(path_prefix, p) for p in df['frame01']],
                         'mov':[os.path.join(path_prefix, p) for p in df['frame02']]}
            seg_files = {'fix': [os.path.join(path_prefix, p) for p in df['frame01_gt']],
                         'mov': [os.path.join(path_prefix, p) for p in df['frame02_gt']]}

    # if data == 'prostate':
    #     img_files = [os.path.join(path_prefix,p) for p in df['t2w']]
    # elif data == 'miami_prostate':
    #     img_files = [os.path.join(path_prefix,p) for p in df['t2-0']]

    # seg_files = [os.path.join(path_prefix,p) for p in df['prostate_mask']]
    # seg_files = [os.path.join(path_prefix,p) for p in df[task]]


    # img_files = [os.path.join(path_prefix, d[i][0].strip("/")) for i in list(d.keys())]
    # seg_files = [os.path.join(path_prefix, d[i][1].strip("/")) for i in list(d.keys())]

    dataset = DATASET_DICT[data](
        img_files,
        seg_files,
        batch_size=batch_size,
        split=split,
        augmentation=augmentation,
        rand_crop_spatial_size=rand_crop_spatial_size,
        convert_to_sam=convert_to_sam,
        do_test_crop=do_test_crop,
        do_nnunet_intensity_aug=do_nnunet_intensity_aug,
        multi_class=multi_class,
    )

    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_worker, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=num_worker, drop_last=True
        )
    return loader

