import os
import glob
import cv2
import multiprocessing
import numpy as np
import torch


## Resize and shift the dynamic range of image to 0-1
def resize_fid_image(img):
    """
    Args:
        img(np.array): shape (N, H, W) and dtype float32 between [0,1] or np.uint8
    Return:
        img(torch.tensor): shape (3, 299, 299) and dtype torch.float32 between [0,1]
    """
    assert img.shape[2] == 3
    assert len(img.shape) == 3

    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255
    img = cv2.resize(img, (299, 299))
    img = np.rollaxis(img, axis=2)
    img = toch.from_numpy(img)

    assert img.max() <= 1.0
    assert img.min() >= 0.0
    assert img.dtype == torch.float32
    assert img.shape == (3, 299, 299)

    return img

## Resize and shift the dynamic range of images to 0-1
def resize_fid_images(imgs, use_multi_processing=True):
    if use_multi_processing:
        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            jobs = []
            for img in imgs:
                job = pool.apply(resize_fid_image, (img,))
                jobs.append(job)
            final_imgs = torch.zeros(imgs.shape[0], 3, 299, 299)
            for idx, job in enumerate(jobs):
                img = job.get()
                final_imgs[idx] = img
    else:
        final_imgs = torch.stack([resize_fid_image(img) for img in imgs], dim=0)
    
    assert final_imgs.shape == (imgs.shape[0], 3, 299, 299)
    assert final_imgs.max() <= 1.0
    assert final_imgs.min() >= 0.0
    assert final_imgs.dtype == torch.float32

    return final_imgs

## Load all images(.png / .jpg) from a given path. All images must be same dtype and shape
def load_images(img_path):
    img_paths = []
    img_extensions = ['png', 'jpg']
    for ext in img_extensions:
        print("Checking for images in ", os.path.join(img_path, "*.{}".format(ext)))
        for img_path in glob.glob(os.path.join(img_path, "*.{}".format(ext))):
            img_paths.append(img_path)
    
    first_img = cv2.imread(img_paths[0])
    W, H = first_img.shape[:2]
    img_paths.sort()
    final_imgs = np.zeros((len(img_paths), H, W, 3), dtype=first_img.dtype)

    for idx, img_path in enumerate(img_paths):
        img = cv2.imread(img_path)
        img = img[:, :, ::-1] # BRG => RGB
        assert img.dtype == final_imgs.dtype
        final_imgs[idx] = img
    
    return final_imgs