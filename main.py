import argparse
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from model import *
import config

def main():
    parser = argparse.ArgumentParser(description='Loss functions for Generative Adversarial Networks', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset', required=True, help='mnist', dest='dataset')
    parser.add_argument('--batch_size', type=int, default=64, dest='batch_size')
    parser.add_argument('--img_size', type=int, default=64, dest='img_size')
    parser.add_argument('--num_epoch', type=int, default=100, dest='num_epoch')
    parser.add_argument('--lr', type=float, default=2e-4, dest='lr')
    parser.add_argument('--beta', type=float, default=0.5, dest='beta')

    parser.add_argument('--loss_type', type=str, default='gan', dest='loss_type')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # data loader

    # loss
    criterion = nn.BCELoss()

    # train mode
    dataset_transforms = transforms.Compose([
        transforms.Scale(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize(config.mean[args.dataset], config.std[args.dataset])
    ])
    pass


if __name__ == '__main__':
    main()