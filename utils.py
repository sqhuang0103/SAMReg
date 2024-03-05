import os
import numpy as np
import nibabel as nib
from scipy.spatial.distance import cdist
class Metric():
    def __init__(self):
        self.reset()

    def reset(self):
        self.dice_sum = 0.0
        self.hd_sum = 0.0
        self.tre_sum = 0.0
        self.count = 0
        self.tre_list = []
        self.dice_list = []

    def dice_coefficient(self, mask1, mask2):
        """calculate Dice"""
        intersection = np.logical_and(mask1, mask2).sum()
        return 2. * intersection / (mask1.sum() + mask2.sum())

    def hausdorff_distance_95(self,mask1, mask2):
        """calculate TRE 95% Hausdorff distance"""
        coords1 = np.argwhere(mask1)
        coords2 = np.argwhere(mask2)

        distances = cdist(coords1, coords2, metric='euclidean')

        hd_95 = np.percentile(np.hstack([distances.min(axis=0), distances.min(axis=1)]), 95)
        return hd_95

    def target_registration_error(self, mask1, mask2):
        """calculate TRE"""
        # 计算每个掩模的前景中点
        centroid1 = np.mean(np.argwhere(mask1), axis=0)
        centroid2 = np.mean(np.argwhere(mask2), axis=0)
        # 计算两个中点之间的欧氏距离
        tre = np.linalg.norm(centroid1 - centroid2)
        return tre

    def update(self, mask1, mask2):
        dice = self.dice_coefficient(mask1, mask2)
        tre = self.target_registration_error(mask1, mask2)
        # hd_95 = self.hausdorff_distance_95(mask1, mask2)
        self.dice_sum += dice
        self.tre_sum += tre
        # self.hd_sum += hd_95
        self.tre_list.append(tre)
        self.dice_list.append(dice)
        self.count += 1

    def get_dice(self):
        if self.count == 0:
            return 0
        mean_dice = np.mean(self.dice_list)
        std_deviation = np.std(self.dice_list)
        return mean_dice, std_deviation
        # return self.dice_sum / self.count

    def get_hd_95(self):
        if self.count == 0:
            return 0
        return self.hd_sum / self.count

    def get_tre(self):
        if self.count == 0:
            return 0
        mean_tre = np.mean(self.tre_list)
        std_deviation = np.std(self.tre_list)
        return mean_tre, std_deviation
        # return self.tre_sum / self.count

def write_nii_data(save_path,file_name,data):
    # data: nd.array
    # 将 NumPy 数组转换为 NIfTI 图像对象
    nifti_image = nib.Nifti1Image(data, np.eye(4))

    # 保存为 nii.gz 文件
    save_path = os.path.join(save_path,file_name)
    nib.save(nifti_image, save_path)
    # print(f"Saved the NIfTI image to {save_path}")