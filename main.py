import argparse
import time

import torch
import torch.nn as nn
import torchvision

from torch.utils.data import DataLoader
from torchvision import transforms

from model import *
import config

def main():
    parser = argparse.ArgumentParser(description='Loss functions for Generative Adversarial Networks', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset', required=True, help='mnist', dest='dataset')
    parser.add_argument('--batch_size', type=int, default=64, dest='batch_size')
    parser.add_argument('--num_epoch', type=int, default=100, dest='num_epoch')
    parser.add_argument('--lr', type=float, default=2e-4, dest='lr')
    parser.add_argument('--beta', type=float, default=0.5, dest='beta')

    parser.add_argument('--loss_type', type=str, default='gan', dest='loss_type')
    
    parser.add_argument('--img_size', type=int, default=64, dest='img_size')
    parser.add_argument('--num_channel', type=int, default=3, dest='num_channel')
    parser.add_argument('--nker', type=int, default=128, dest='nker')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # data loader
    dataset_transforms = transforms.Compose([
        transforms.Scale(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize(config.mean[args.dataset], config.std[args.dataset])
    ])

    
    if args.dataset == 'mnist':
        dataset = torchvision.datasets.MNIST(
            root='./dataset/mnist/',
            download=True,
            transform=dataset_transforms
        )
    elif args.dataset == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(
            root='./dataset/cifar10/',
            download=True,
            transform=dataset_transforms
        )
    else:
        print("No Dataset")
    
    dataset_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True)

    # models
    netG = DCGANgenrator(in_channels=100, out_channels=args.num_channel, nker=args.nker).to(device)
    netD = DCGANdiscriminator(in_channels=args.num_channel, out_channels=1, nker=args.nker).to(device)


    # loss & optim
    criterion = nn.BCELoss()
    optimG = torch.optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta, 0.999))
    optimD = torch.optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beata,0.999))

    
    # global variables
    noise = torch.autograd.Variable(torch.FloatTensor(args.batch_size, 100, 1, 1))
    real = torch.autograd.Variable(torch.FloatTensor(args.batch_size, args.num_channel, args.img_size. args.img_size))
    label = torch.autograd.Variable(torch.FloatTensor(args.batch_size))
    real_label, fake_label = 1, 0

    start_time = time.time()

    for epoch in range(1, num_epoch + 1):
        for bach, data in enumerate(dataset_loader, 1):
            # Gradient of Discriminator
            netD.zero_grad()

            # REAL DATA
            real.data.resize_(data.size()).copy_(data)
            label.data.resize_(data.size(0)).fill_(real_label)

            output = netD(real)
            loss_d_real = 0.5 * torch.mean((output - label)**2)
            loss_d_real.backward()

            # FAKE DATA
            label.data.fill_(fake_label)
            noise.data.resize_(data.size(0), 100, 1, 1)
            noise.data.normal_(0, 1)

            fake = netG(noise)
            output = netD(fake.detach())
            loss_d_fake = 0.5 * torch.mean((output - label)**2)
            loss_d_fake.backward()

            loss_d = loss_d_fake + loss_d_real
            optimD.step()

            # Gradient of Generator
            netG.zero_grad()
            label.data.fill_(real_label)
            output = netD(fake)
            loss_g = 0.5 * torch.mean((output - label)**2)
            loss_g.backward()
            optimG.step()

        # savw
        if epoch % 2 == 0:
            pass


    end_time = time.time()
    print('Start to End Time: %f' % (end_time - start_time))
    


if __name__ == '__main__':
    main()