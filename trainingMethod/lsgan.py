import torch

def train_lsgan_one_epoch(netG, netD, optimG, optimD, batch_size, num_channels, img_size, dataset_loader):
    # global variables
    noise = torch.autograd.Variable(torch.FloatTensor(batch_size, 100, 1, 1))
    real = torch.autograd.Variable(torch.FloatTensor(batch_size, num_channels, img_size, img_size))
    label = torch.autograd.Variable(torch.FloatTensor(batch_size))
    real_label, fake_label = 1, 0

    for batch, data in enumerate(dataset_loader, 1):
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

        """print('TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | G_LOSS: %.4f | D_LOSS: %.4f'
            % (epoch, args.num_epoch, batch, args.batch_size, loss_g, loss_d))"""
        print('BATCH %04d / %04d | G_LOSS: %.4f | D_LOSS: %.4f'
            % (batch, batch_size, loss_g, loss_d))