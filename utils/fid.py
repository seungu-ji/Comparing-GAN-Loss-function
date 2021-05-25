## Compute Frechet Inception Distance
import os
import numpy as np
import cv
from scipy import linalg
from optparse import OptionParser

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import inception_v3

from utils.utils import *

def to_cuda(elements):
    """
    Transfers elements to cuda if GPU is available
    Args:
        elements: torch.tensor or torch.nn.module
        --
    Returns:
        elements: same as input on GPU memory, if available
    """
    if torch.cuda.is_available():
        return elements.cuda()

    return elements

class PartialInceptionNetwork(nn.Module):
    def __init__(self, transform_input=True):
        super().__init__()
        self.inception_network = inception_v3(pretrained=True)
        """
        In torchvision Inception3, the Mixed_&c layer is the previous layer of avg_pool2d
        and output is (N x 2048 x 8 x 8), where N is batch
        """
        self.inception_network.Mixed_7c.register_forward_hook(self.output_hook)
        self.transform_input = transform_input

    def output_hook(self, module, input, output):
        # N x 2048 x 8 x 8
        self.mixed_7c_ouput = output

    def forward(self, x):
        """ 
        Args:
            x(torch.float32): shape (N, 3, 299, 299), each element is range[0,1]
        Return:
           inception(torch.float32): inception activations (N, 2048) 
        """
        assert x.shape[1:] == (3, 299, 299), "Expected input shape to be: (N, 3, 299, 299)" +\
                                             "but, got input shape {}".fotmat(x.shape)

        # normalize to [-1, 1]
        x = x * 2 - 1 

        # Trigger output hook
        self.inception_network(x)

        # Output: N x 2048 x 1 x 1
        activations = self.mixed_7c_ouput
        activations = F.adaptive_avg_pool2d(activations, (1,1))
        activations = activations.view(x.shape[0], 2048)

        return activations

## Calculates activations for last pool layer for all images
def get_activations(imgs, batch_size, device=None):
    """
    Args:
        img(torch.array): shape(N, 3, 299, 299)
        batch_size(int): batch size used for inception network
        device: whether to use cuda
    Return:
        inception_activations(np.float32): shape (N, 2048)
    """
    assert imgs.shape[1:] == (3, 299, 299), "Expected input shape to be: (N, 3, 299, 299)" +\
                                           "but, got input shape {}".format(imgs.shape)

    num_imgs = imgs.shape[0]
    inception_network = PartialInceptionNetwork()
    if device is not None:
        inception_network.to(device)
    inception_v3.eval()
    num_batch = int(np.ceil(num_imgs / batch_size))
    inception_activations = np.zeros((num_imgs, 2048), dtype=np.float32)

    for batch_idx in range(num_batch):
        start_idx = batch_size * batch_idx
        end_idx = start_idx + batch_size

        img = imgs[start_idx:end_idx]
        img = img.to(device)
        activations = inception_network(img)
        activations = activations.detach().cpu().numpy()
        assert activations.shape == (img.shape[0], 2048), "Expected output shape to be: {}" +\
                                                          "but, got {}".format((img.shape[0], 2048), activations.shape)
        inception_activations[start_idx:end_idx, :] = activations

    return inception_activations

## Calcalates the statistics(mu, sigma) used by FID
def calculate_activation_statistics(imgs, batch_size):
    """
    Args:
        imgs(torch.float32): shape(N, 3, H, W)
        batch_size: batch_size to use to calculate inception scores
    Return:
        mu: mean over all activations from the last pool layer of the inception model
        sigma: covariance matrix over all activations from the last pool layer of the inception model
    """
    act = get_activations(imgs, batch_size)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)

    return mu, sigma

## Numpy implementation of the Frechet Distance
def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Args:
        mu1: The mean over activations of the pool_3 layer of the inception v3 for fake samples(generated samples)
        mu2: The mean over activations of the pool_3 layer of the inception v3 for real samples
        sigma1: The covariance matrix over activations of the pool_3 layer for fake samples(generated samples)
        sigam2: THe covariance matrix over activations of the pool_3 layer for real samples
        eps: The epsilon value
    Return:
        frechet_distance(float): The Frechet Distance
    """
    mu1, mu2 = np.atleast_1d(mu1), np.atleast_1d(mu2)
    sigma1, sigam2 = np.atleast_2d(sigma1), np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and Test mean vectors have different lenghs"
    assert sigma1.shape == sigma2.shape, "Training and Test covariances haver different dimensions"

    diff = mu1 - mu2
    # product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigam2), disp=False)
    if not np.isfinite(covmean).all():
        warning.warn("FID calculation produces singular product" +\
                     "adding %s to diagonal of cov estimates" % eps)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((simga1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    frechet_distance = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)

    return frechet_distance

## Calculate FID between images1 and images2
def calculate_fid(imgs1, imgs2, use_multi_processing, batch_size):
    """
    Args:
        imgs1(np.array): shape (N, H, W, 3) and dtype np.float32 between [0,1] or np.uint8
        imgs2(np.array): shape (N, H, W, 3) and dtype np.float32 between [0,1] or np.uint8
        use_multi_processing(bool): whether to use multiprocessing
        batch_size(int): batch size used for inception network
    Return:
        fid(scalar): Frechet Inception Distance
    """
    imgs1 = resize_fid_images(imgs1, use_multi_processing)
    imgs2 = resize_fid_images(imgs2, use_multi_processing)
    mu1, sigma1 = calculate_activation_statistics(imgs1, batch_size)
    mu2, sigma2 = calculate_activation_statistics(imgs2, batch_size)
    fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)

    return fid

if __name__ == 'main':
    parser = OptionParser()
    parser.add_option("--p1", "--path1", dest="path1", 
                      help="Path to directory containing the real images")
    parser.add_option("--p2", "--path2", dest="path2", 
                      help="Path to directory containing the generated images")
    parser.add_option("--multiprocessing", dest="use_multiprocessing",
                      help="Toggle use of multiprocessing for image pre-processing. Defaults to use all cores",
                      default=False,
                      action="store_true")
    parser.add_option("-b", "--batch-size", dest="batch_size",
                      help="Set batch size to use for InceptionV3 network",
                      type=int)
    
    options, _ = parser.parse_args()
    assert options.path1 is not None, "--path1 is an required option"
    assert options.path2 is not None, "--path2 is an required option"
    assert options.batch_size is not None, "--batch_size is an required option"
    images1 = load_images(options.path1)
    images2 = load_images(options.path2)
    fid_value = calculate_fid(images1, images2, options.use_multiprocessing, options.batch_size)
    print(fid_value)