import os
import nibabel as nib
from scipy.spatial.distance import cdist
import matplotlib.colors as mcolors
import requests
import matplotlib.pyplot as plt
import random
import numpy as np
import cv2
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
        centroid1 = np.mean(np.argwhere(mask1), axis=0)
        centroid2 = np.mean(np.argwhere(mask2), axis=0)

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

            # image1_with_mask = self.overlay_mask_on_image(image1, mask1['segmentation'], random_color)
            image1_with_mask = self.overlay_mask_on_image(image1, mask1, random_color)
            # image2_with_mask = self.overlay_mask_on_image(image2, mask2['segmentation'], random_color)
            image2_with_mask = self.overlay_mask_on_image(image2, mask2, random_color)

            # Concatenate images horizontally
            combined_images = np.hstack((image1_with_mask, image2_with_mask))

            # Display combined images
            cv2.imshow(f'Image Pair {ind}', combined_images)
            # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    def overlay_merge_mask_on_image(self,image, mask, color):
        """
        Applies a mask to an image using a specified color.
        """
        # Create an overlay with the same size as the image, filled with the specified color
        overlay = np.zeros_like(image)
        mask_indices = mask > 0
        overlay[mask_indices] = color  # Apply color where mask is true

        # Blend the overlay with the original image
        image_with_mask = cv2.addWeighted(image, 0.5, overlay, 0.5, 0)
        return image_with_mask

    def _show_merge_cor_img(self, image1, image2, masks1, masks2):
        """
        Displays images with all masks merged onto them.
        """
        # Copy the original images to keep them unchanged
        image1_all_masks = image1.copy()
        image2_all_masks = image2.copy()

        # Apply each mask to the copy of the original images
        for mask1, mask2 in zip(masks1, masks2):
            random_color = np.random.randint(0, 256, (3,), dtype=np.uint8)
            image1_all_masks = self.overlay_merge_mask_on_image(image1_all_masks, mask1, random_color)
            image2_all_masks = self.overlay_merge_mask_on_image(image2_all_masks, mask2, random_color)

        # Concatenate the images with all masks applied
        combined_images = np.hstack((image1_all_masks, image2_all_masks))


            # Display combined images
        cv2.imshow('Image Pair with All Masks', combined_images)
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
    nifti_image = nib.Nifti1Image(data, np.eye(4))

    save_path = os.path.join(save_path,file_name)
    nib.save(nifti_image, save_path)
    # print(f"Saved the NIfTI image to {save_path}")

#======================================================================================

def visualize_masks(image1, masks1, image2, masks2):
    # Convert PIL images to numpy arrays
    background1 = np.array(image1)
    background2 = np.array(image2)

    # Convert RGB to BGR (OpenCV uses BGR color format)
    background1 = cv2.cvtColor(background1, cv2.COLOR_RGB2BGR)
    background2 = cv2.cvtColor(background2, cv2.COLOR_RGB2BGR)

    # Create a blank mask for each image
    mask1 = np.zeros_like(background1)
    mask2 = np.zeros_like(background2)

    distinct_colors = [
        (0, 255, 0),  # Green
        (0, 0, 255),  # Blue
        (255, 0, 0),  # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 0, 0),  # Maroon
        (0, 128, 0),  # Olive
        (0, 0, 128),  # Navy
        (128, 128, 0),  # Teal
        (128, 0, 128),  # Purple
        (0, 128, 128),  # Gray
        (192, 192, 192),  # Silver
        (0, 128, 128), (216, 145, 38), (211, 38, 202), (42, 216, 51), (109, 38, 216), (38, 118, 216), (216, 38, 91),
        (216, 51, 42), (38, 189, 216), (192, 192, 192), (163, 216, 38), (48, 48, 216), (0, 0, 128), (38, 216, 109),
        (128, 0, 0), (255, 255, 0), (100, 216, 38), (128, 128, 0), (0, 128, 0), (0, 255, 255), (128, 0, 128),
        (255, 0, 255), (0, 255, 0), (211, 202, 38), (255, 0, 0), (216, 38, 145), (0, 0, 255), (38, 216, 171),
        (216, 100, 38), (163, 38, 216)


    ]

    def random_color():
        """Generate a random color with high saturation and value in HSV color space."""
        hue = random.randint(0, 179)  # Random hue value between 0 and 179 (HSV uses 0-179 range)
        saturation = random.randint(200, 255)  # High saturation value between 200 and 255
        value = random.randint(200, 255)  # High value (brightness) between 200 and 255
        color = np.array([[[hue, saturation, value]]], dtype=np.uint8)
        return cv2.cvtColor(color, cv2.COLOR_HSV2BGR)[0][0]


    # Iterate through mask lists and overlay on the blank masks with different colors
    for idx, (mask1_item, mask2_item) in enumerate(zip(masks1, masks2)):
        # color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        color = distinct_colors[idx % len(distinct_colors)]
        # color = random_color()
        # Convert binary masks to uint8
        mask1_item = np.uint8(mask1_item)
        mask2_item = np.uint8(mask2_item)

        # Create a mask where binary mask is True
        fg_mask1 = np.where(mask1_item, 255, 0).astype(np.uint8)
        fg_mask2 = np.where(mask2_item, 255, 0).astype(np.uint8)

        # Apply the foreground masks on the corresponding masks with the same color
        mask1[fg_mask1 > 0] = color
        mask2[fg_mask2 > 0] = color

    # Add the masks on top of the background images
    result1 = cv2.addWeighted(background1, 1, mask1, 0.5, 0)
    result2 = cv2.addWeighted(background2, 1, mask2, 0.5, 0)

    cv2.imshow('fix_result',result1)
    cv2.imshow('mov_result',result2)

    return result1, result2

def visualize_masks_with_scores(image, masks, scores, points):
    """
    Visualize masks with their scores on the original image.

    Parameters:
    - image: PIL image with size (H, W)
    - masks: torch tensor of shape [1, 3, H, W]
    - scores: torch tensor of scores with shape [1, 3]
    """
    # Convert PIL image to NumPy array
    image_np = np.array(image)

    # Move masks and scores to CPU and convert to NumPy
    masks_np = masks.cpu().numpy().squeeze(0)  # Shape [3, H, W]
    scores_np = scores.cpu().numpy().squeeze(0)  # Shape [3]

    # Set up the plot
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    for i in range(3):
        ax = axs[i]
        score = scores_np[i]
        mask = masks_np[i]
        # Create an RGBA image for the mask
        mask_image = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
        mask_image[mask] = [255, 0, 0, 255]
        # mask_image[..., 3] = mask * 255  # Alpha channel
        # Overlay the mask on the image
        ax.imshow(image_np)
        # ax.imshow(mask_image, cmap='Reds', alpha=0.5)
        ax.imshow(mask_image, alpha=0.5)
        ax.scatter(points[:, 0], points[:, 1], c='red', marker='o', label='Scatter Points')
        ax.set_title(f'Score: {score:.4f}')
        ax.axis('off')
    plt.tight_layout()
    # plt.show()

def visualize_masks_with_sim(image, masks):
    """
    Visualize masks with their scores on the original image.

    Parameters:
    - image: PIL image with size (H, W)
    - masks: torch tensor of shape [1, 3, H, W]
    - scores: torch tensor of scores with shape [1, 3]
    """
    # Convert PIL image to NumPy array
    image_np = np.array(image)

    # Move masks and scores to CPU and convert to NumPy

    masks = [m.cpu().numpy() for m in masks]  # Shape [3, H, W]
    masks = [m.astype('uint8') for m in masks]
    masks_np = np.array(masks)

    # Set up the plot
    fig, axs = plt.subplots(1, masks_np.shape[0], figsize=(15, 5))
    for i in range(masks_np.shape[0]):
        ax = axs[i]
        mask = masks_np[i]
        # Create an RGBA image for the mask
        mask_image = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
        mask_image[mask>0] = [255, 0, 0, 255]

        # mask_image[..., 3] = mask * 255  # Alpha channel
        # Overlay the mask on the image
        ax.imshow(image_np)
        ax.imshow(mask_image, cmap='Reds', alpha=0.5)

        ax.axis('off')
    plt.tight_layout()
    # plt.show()

def create_transparent_mask(binary_mask, save_path, foreground_color=(12, 34, 234), alpha=0.5):
    """
    Convert a binary mask to a colorful transparent mask using OpenCV.

    Args:
    binary_mask (numpy.array): A binary mask of shape (1, h, w)
    foreground_color (tuple): RGB color for the mask foreground
    alpha (float): Alpha transparency value

    Returns:
    numpy.array: An RGBA image as a numpy array
    """
    # Check input dimensions
    if binary_mask.shape[0] != 1:
        raise ValueError("Expected binary mask with shape (1, h, w)")
    binary_mask = np.uint8(binary_mask>0)

    # Remove the first dimension and create an RGB image based on the binary mask
    mask_rgb = np.zeros((*binary_mask.shape[1:], 3), dtype=np.uint8)
    mask_rgb[binary_mask[0] == 1] = foreground_color

    # Create an alpha channel based on the binary mask
    mask_alpha = (binary_mask[0] * alpha * 255).astype(np.uint8)

    # Combine the RGB and alpha channels to create an RGBA image
    mask_rgba = cv2.merge((mask_rgb[:, :, 0], mask_rgb[:, :, 1], mask_rgb[:, :, 2], mask_alpha))
    cv2.imwrite(save_path,mask_rgba)

    return mask_rgba





