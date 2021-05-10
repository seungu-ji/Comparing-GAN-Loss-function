import torch

def train_wgan_one_epoch(netG, netD, optimG, optimD, num_critic, weight_clip, batch_size, num_channels, img_size, dataset_loader):
    for batch, data in enumerate(dataset_loader, 1):
        real = data['label']#.to(device)
        noise = torch.randn(real.shape[0], 100, 1, 1,)#.to(device) # (B, C, H, W)
        fake = netG(noise) # Generate Fake Image


        # Gradient of Discriminator
        optimD.zero_grad()

        # Adversarial Loss
        loss_d = -torch.mean(netD(real)) + torch.mean(netD(fake.detach()))
        loss_d.backward()
        optimD.step()

        # Clip weight of discriminator
        for p in netD.parameters():
            p.data.clamp_(-weight_clip, weight_clip)

        # Train the generator every num_critic iterations
        if batch % num_critic == 0:
            optimG.zero_grad()
            pred_fake = netD(fake)
            loss_g = -torch.mean(pred_fake)
            loss_g.backward()
            optimG.step()
        

        """print('TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | G_LOSS: %.4f | D_LOSS: %.4f'
            % (epoch, args.num_epoch, batch, args.batch_size, loss_g, loss_d))"""
        print('BATCH %04d / %04d | G_LOSS: %.4f | D_LOSS: %.4f'
            % (batch, batch_size, loss_g, loss_d))