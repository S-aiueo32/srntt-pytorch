import argparse
from pathlib import Path

from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models import SRNTT, Discriminator, ImageDiscriminator
from datasets import ReferenceDataset, ReferenceDatasetEval
from losses import (AdversarialLoss, PerceptualLoss, TextureLoss,
                    PSNR, SSIM, compute_gp)
from utils import init_seeds

torch.autograd.set_detect_anomaly(True)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
init_seeds(seed=123)


def parse_args():
    parser = argparse.ArgumentParser(description='Train SRNTT')
    # data setting
    parser.add_argument('--dataroot', type=str, default='data/CUFED')
    # train setting
    parser.add_argument('--n_epochs_init', type=int, default=5)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=9)
    # model setting
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=32)
    parser.add_argument('--netD', default='image', choices=['patch', 'image'])
    parser.add_argument('--n_blocks', type=int, default=16)
    parser.add_argument('--use_weights', action='store_true')
    # loss function setting
    parser.add_argument('--lambda_rec', type=float, default=1.)
    parser.add_argument('--lambda_per', type=float, default=1e-4)
    parser.add_argument('--lambda_tex', type=float, default=1e-4)
    parser.add_argument('--lambda_adv', type=float, default=1e-6)
    # optimizer setting
    parser.add_argument('--lr', type=float, default=1e-4)
    # logging setting
    parser.add_argument('--pid', type=str, default=None)
    parser.add_argument('--display_freq', type=int, default=100)
    # weights setting
    parser.add_argument('--init_weight', type=str, default='weights/SRGAN.pth')
    parser.add_argument('--pretrain', type=str, default=None)
    # debug
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()


def main(args):
    # split data
    files = list([f.stem for f in Path(args.dataroot).glob('map/*.npz')])
    train_files, val_files = train_test_split(files, test_size=0.1)

    # define dataloaders
    train_set = ReferenceDataset(train_files, args.dataroot)
    val_set = ReferenceDatasetEval(val_files, args.dataroot)
    train_loader = DataLoader(
        train_set, args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, args.batch_size, drop_last=True)

    # define networks
    netG = SRNTT(args.ngf, args.n_blocks, args.use_weights).to(device)
    netG.content_extractor.load_state_dict(torch.load(args.init_weight))
    if args.netD == 'image':
        netD = ImageDiscriminator(args.ndf).to(device)
    elif args.netD == 'patch':
        netD = Discriminator(args.ndf).to(device)

    # define criteria
    criterion_rec = nn.L1Loss().to(device)
    criterion_per = PerceptualLoss().to(device)
    criterion_adv = AdversarialLoss().to(device)
    criterion_tex = TextureLoss(args.use_weights).to(device)

    # metrics
    criterion_psnr = PSNR(max_val=1., mode='Y')
    criterion_ssim = SSIM(window_size=11)

    # define optimizers
    optimizer_G = optim.Adam(netG.parameters(), args.lr)
    optimizer_D = optim.Adam(netD.parameters(), args.lr)

    scheduler_G = StepLR(
        optimizer_G, int(args.n_epochs * len(train_loader) / 2), 0.1)
    scheduler_D = StepLR(
        optimizer_D, int(args.n_epochs * len(train_loader) / 2), 0.1)

    # for tensorboard
    writer = SummaryWriter(log_dir=f'runs/{args.pid}' if args.pid else None)

    if args.pretrain is None:
        """ pretrain """
        step = 0
        for epoch in range(1, args.n_epochs_init + 1):
            for i, batch in enumerate(train_loader, 1):
                img_hr = batch['img_hr'].to(device)
                img_lr = batch['img_lr'].to(device)
                maps = {k: v.to(device) for k, v in batch['maps'].items()}
                weights = batch['weights'].to(device)

                _, img_sr = netG(img_lr, maps, weights)

                """ train G """
                optimizer_G.zero_grad()
                g_loss = criterion_rec(img_sr, img_hr)
                g_loss.backward()
                optimizer_G.step()

                """ logging """
                writer.add_scalar('pre/g_loss', g_loss.item(), step)
                if step % args.display_freq == 0:
                    writer.add_images('pre/img_lr', img_lr.clamp(0, 1), step)
                    writer.add_images('pre/img_hr', img_hr.clamp(0, 1), step)
                    writer.add_images('pre/img_sr', img_sr.clamp(0, 1), step)

                log_txt = [
                    f'[Pre][Epoch{epoch}][{i}/{len(train_loader)}]',
                    f'G Loss: {g_loss.item()}'
                ]
                print(' '.join(log_txt))

                step += 1

                if args.debug:
                    break

            out_path = Path(writer.log_dir) / f'netG_pre{epoch:03}.pth'
            torch.save(netG.state_dict(), out_path)

    else:  # ommit pre-training
        netG.load_state_dict(torch.load(args.pretrain))

    """ train with all losses """
    step = 0
    for epoch in range(1, args.n_epochs + 1):
        """ training loop """
        netG.train()
        netD.train()
        for i, batch in enumerate(train_loader, 1):
            img_hr = batch['img_hr'].to(device)
            img_lr = batch['img_lr'].to(device)
            maps = {k: v.to(device) for k, v in batch['maps'].items()}
            weights = batch['weights'].to(device)

            _, img_sr = netG(img_lr, maps, weights)

            """ train D """
            optimizer_D.zero_grad()
            for p in netD.parameters():
                p.requires_grad = True
            for p in netG.parameters():
                p.requires_grad = False

            # compute WGAN loss
            d_out_real = netD(img_hr)
            d_loss_real = criterion_adv(d_out_real, True)
            d_out_fake = netD(img_sr.detach())
            d_loss_fake = criterion_adv(d_out_fake, False)
            d_loss = d_loss_real + d_loss_fake

            # gradient penalty
            gradient_penalty = compute_gp(netD, img_hr.data, img_sr.data)
            d_loss += 10 * gradient_penalty

            d_loss.backward()
            optimizer_D.step()

            """ train G """
            optimizer_G.zero_grad()
            for p in netD.parameters():
                p.requires_grad = False
            for p in netG.parameters():
                p.requires_grad = True

            # compute all losses
            loss_rec = criterion_rec(img_sr, img_hr)
            loss_per = criterion_per(img_sr, img_hr)
            loss_adv = criterion_adv(netD(img_sr), True)
            loss_tex = criterion_tex(img_sr, maps, weights)

            # optimize with combined d_loss
            g_loss = (loss_rec * args.lambda_rec +
                      loss_per * args.lambda_per +
                      loss_adv * args.lambda_adv +
                      loss_tex * args.lambda_tex)
            g_loss.backward()
            optimizer_G.step()

            """ logging """
            writer.add_scalar('train/g_loss', g_loss.item(), step)
            writer.add_scalar('train/loss_rec', loss_rec.item(), step)
            writer.add_scalar('train/loss_per', loss_per.item(), step)
            writer.add_scalar('train/loss_tex', loss_tex.item(), step)
            writer.add_scalar('train/loss_adv', loss_adv.item(), step)
            writer.add_scalar('train/d_loss', d_loss.item(), step)
            writer.add_scalar('train/d_real', d_loss_real.item(), step)
            writer.add_scalar('train/d_fake', d_loss_fake.item(), step)
            if step % args.display_freq == 0:
                writer.add_images('train/img_lr', img_lr, step)
                writer.add_images('train/img_hr', img_hr, step)
                writer.add_images('train/img_sr', img_sr.clamp(0, 1), step)

            log_txt = [
                f'[Train][Epoch{epoch}][{i}/{len(train_loader)}]',
                f'G Loss: {g_loss.item()}, D Loss: {d_loss.item()}'
            ]
            print(' '.join(log_txt))

            scheduler_G.step()
            scheduler_D.step()

            step += 1

            if args.debug:
                break

        """ validation loop """
        netG.eval()
        netD.eval()
        val_psnr, val_ssim = 0, 0
        tbar = tqdm(total=len(val_loader))
        for i, batch in enumerate(val_loader, 1):
            img_hr = batch['img_hr'].to(device)
            img_lr = batch['img_lr'].to(device)
            maps = {k: v.to(device) for k, v in batch['maps'].items()}
            weights = batch['weights'].to(device)

            with torch.no_grad():
                _, img_sr = netG(img_lr, maps, weights)
                val_psnr += criterion_psnr(img_hr, img_sr.clamp(0, 1)).item()
                val_ssim += criterion_ssim(img_hr, img_sr.clamp(0, 1)).item()

            tbar.update(1)

            if args.debug:
                break
        else:
            tbar.close()
            val_psnr /= len(val_loader)
            val_ssim /= len(val_loader)

        writer.add_scalar('val/psnr', val_psnr, epoch)
        writer.add_scalar('val/ssim', val_ssim, epoch)

        print(f'[Val][Epoch{epoch}] PSNR:{val_psnr:.4f}, SSIM:{val_ssim:.4f}')

        netG_path = Path(writer.log_dir) / f'netG_{epoch:03}.pth'
        netD_path = Path(writer.log_dir) / f'netD_{epoch:03}.pth'
        torch.save(netG.state_dict(), netG_path)
        torch.save(netD.state_dict(), netD_path)


if __name__ == "__main__":
    args = parse_args()
    main(args)
