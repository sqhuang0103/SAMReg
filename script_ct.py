import glob

import nibabel as nib
import matplotlib.pyplot as plt
import cv2
import numpy as np

# scan_nii_path = '/home/shiqi/3DSAM/sam_reg/CT_example/scans/case_001_exp.nii.gz'
# scan_nii_path = 'amos_abd_ct/scan/amos_0007.nii.gz'
# mask_nii_path = '/home/shiqi/3DSAM/sam_reg/CT_example/masks/case_001_exp.nii.gz'
def read_nii_scan(nii_path):
    img = nib.load(nii_path)
    data = img.get_fdata()
    print('shape of this scan: {}.'.format(data.shape))
    # print(data.max())
    return data
def read_nii_mask(nii_path,lb=1):
    img = nib.load(nii_path)
    data = img.get_fdata()
    data = np.uint8(data==lb)
    # data = data*255.
    # print(data.max())
    return data

def show_slices(slices):
   """ Function to display row of image slices """
   fig, axes = plt.subplots(1, len(slices))
   for i, slice in enumerate(slices):
       axes[i].imshow(slice.T, cmap="gray", origin="lower")

def over_vis(nii_path):

    # Select slices from each dimension
    data = read_nii_scan(nii_path)
    print(data.shape) # (192,192,208)
    slice_0 = data[data.shape[0] // 2, :, :]
    slice_1 = data[:, data.shape[1] // 2, :]
    slice_2 = data[:, :, data.shape[2] // 2]

    show_slices([slice_0, slice_1, slice_2])

def vis(nii_path,idx, lb=1, scan=True, save=False):
    if scan:
        data = read_nii_scan(nii_path)
    else:
        data = read_nii_mask(nii_path,lb=lb)
    slice = data[:,:,idx]
    plt.figure()
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.imshow(slice.T, cmap="gray", origin="lower")
    if save:
        if scan:
            plt.savefig("/home/shiqi/"+"lung_scan_a2_s{}.png".format(idx),bbox_inches='tight')
        else:
            plt.savefig("/home/shiqi/"+"lung_mask_a2_s{}.png".format(idx),bbox_inches='tight')

def _vis_lung():
    scan_nii_path = '/home/shiqi/3DSAM/sam_reg/CT_example/scans/case_001_insp.nii.gz'
    # scan_nii_path = 'amos_abd_ct/scan/amos_0007.nii.gz'
    mask_nii_path = '/home/shiqi/3DSAM/sam_reg/CT_example/masks/case_001_insp.nii.gz'

    idx = 125
    for _i in range(0,201,20):
        idx = _i
        vis(scan_nii_path,idx,scan=True,save=True)
        vis(mask_nii_path,idx,scan=False, save=True)
    plt.show()

def _vis_abd():
    scan_nii_path = 'amos_abd_ct/scan/amos_0007.nii.gz'
    mask_nii_path = 'amos_abd_ct/mask/amos_0007.nii.gz'

    idx = 270
    vis(scan_nii_path, idx, scan=True, save=True)
    # for i in range(0,15):
    #     vis(mask_nii_path, idx, lb=i, scan=False, save=True)
    vis(mask_nii_path, idx, lb=2, scan=False, save=True)
    plt.show()

def _vis_prostate():
    scan_nii_path = 'prostate_mri/scan/Patient001061633_study_0.nii.gz'
    mask_nii_path = 'prostate_mri/mask/Patient001061633_study_0.nii.gz'
    idx = 20
    vis(scan_nii_path, idx, scan=True, save=True)
    vis(mask_nii_path, idx, lb=1, scan=False, save=True)
    plt.show()

def _vis_cardiac():
    scan_nii_path = 'cardiac_mri/patient067/patient067_frame10.nii.gz'
    mask_nii_path = 'cardiac_mri/patient067/patient067_frame10_gt.nii.gz'
    idx = 10
    for _i in range(10):
        idx = _i
        vis(scan_nii_path, idx, scan=True, save=True)
        vis(mask_nii_path, idx, lb=1, scan=False, save=True)
    plt.show()

# _vis_abd()
# _vis_prostate()
# _vis_cardiac()
_vis_lung()
##################################################
# 去除保存图片的白色边框
def read_cv2_png(path):
    img = cv2.imread(path)
    return img

def test():
    img = read_cv2_png("/home/shiqi/abd1_scan_a1_s270.png")
    s = img.shape
    img_crop = img[int(s[0] * 0.06):int(s[0] * 0.94), int(s[1] * 0.06):int(s[1] * 0.94), :]
    # img_crop = img[135:346,29:440,:] # a1 cropping para.
    # img_crop = img[11:488,11:488,:] # a2 cropping para.
    cv2.imshow('im', img)
    cv2.imshow('im_crop', img_crop)

name = "/home/shiqi/lung_*_a2_s*.png"
# name = "/home/shiqi/prostate_*_a*_s*.png"
img_paths = glob.glob(name)
print(img_paths)
for img_path in img_paths:
    img = read_cv2_png(img_path)
    s = img.shape
    img_crop = img[int(s[0]*0.06):int(s[0]*0.94),int(s[1]*0.06):int(s[1]*0.94),:]
    cv2.imwrite(img_path, img_crop)
