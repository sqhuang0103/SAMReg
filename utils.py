import os
import numpy as np
import cv2
import nibabel as nib
from scipy.spatial.distance import cdist
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
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

class Vis():
    def __init__(self):
        pass

    def generate_distinct_colors(self,n):
        """implementation to generate fistinct colors"""
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
        """
        Overlays a mask onto an image using a specified color.
        """
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

    def _show_cor_img(self, image1, image2, masks1, masks2):
        """
        Displays pairs of images and their masks.
        """
        # colors = self.generate_distinct_colors(20)
        for ind in range(len(masks1)):
            plt.figure(ind)
            # color = np.random.rand(3,)

            plt.subplot(1, 2, 1)
            plt.imshow(image1.astype('uint8'))
            self.show_mask(masks1[ind]['segmentation'], plt.gca(), random_color=True)
            # self.save_mask(masks1[ind]['segmentation'], plt.gca(), color=colors[ind])
            plt.axis('off')


            plt.subplot(1, 2, 2)
            plt.imshow(image2.astype('uint8'))
            self.show_mask(masks2[ind]['segmentation'], plt.gca(), random_color=True)
            # self.save_mask(masks2[ind]['segmentation'], plt.gca(), color=colors[ind])
            plt.axis('off')

    def _show_interpolate_img(self, moving, moved, fixed):
        """
        Displays the moving image, moved image, and fixed image.
        """
        plt.figure()
        plt.subplot(1,3,1)
        plt.imshow((moving.astype('uint8')*255))
        plt.title('moving image')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow((moved.astype('uint8') * 255))
        plt.title('moving image')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow((fixed.astype('uint8') * 255))
        plt.title('moving image')
        plt.axis('off')

class Vis_cv2():
    def overlay_mask_on_image(self, image, mask, color=None):
        """
        Overlays a mask onto an image using a specified color.
        """
        if color is None:
            color = np.random.randint(0, 256, (3,), dtype=np.uint8)
        image_copy = image.copy()
        mask_indices = np.where(mask != 0)
        image_copy[mask_indices[0], mask_indices[1], :] = color
        return image_copy

    def _show_cor_img(self, image1, image2, masks1, masks2):
        """
        Displays pairs of images and their masks.
        """
        for ind, (mask1, mask2) in enumerate(zip(masks1, masks2)):
            random_color = np.random.randint(0, 256, (3,), dtype=np.uint8)

            image1_with_mask = self.overlay_mask_on_image(image1, mask1['segmentation'], random_color)
            image2_with_mask = self.overlay_mask_on_image(image2, mask2['segmentation'], random_color)

            # Concatenate images horizontally
            combined_images = np.hstack((image1_with_mask, image2_with_mask))

            # Display combined images
            cv2.imshow(f'Image Pair {ind}', combined_images)
            # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    def _show_interpolate_img(self, moving, moved, fixed):
        """
        Displays the moving image, moved image, and fixed image.
        """
        # Concatenate images horizontally
        combined_images = np.hstack((moving, moved, fixed))
        # Display combined images
        cv2.imshow(f'moving - moved - fixed image', combined_images.astype('uint8') * 255)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


def write_nii_data(save_path,file_name,data):
    # data: nd.array
    # 将 NumPy 数组转换为 NIfTI 图像对象
    nifti_image = nib.Nifti1Image(data, np.eye(4))

    # 保存为 nii.gz 文件
    save_path = os.path.join(save_path,file_name)
    nib.save(nifti_image, save_path)
    # print(f"Saved the NIfTI image to {save_path}")
