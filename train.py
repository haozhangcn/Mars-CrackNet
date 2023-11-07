import random
import sys
import os

import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import logging

from models import UNet, Dis, Res_UNet

from data import Mars

from torch.utils import data
from utils import dice_loss, FocalLoss
import numpy as np
from predict import predict
import time
import argparse

timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))

root = os.path.split(os.path.realpath(sys.argv[0]))[0]

parser = argparse.ArgumentParser(description='Mars-CrackNet')
parser.add_argument('-v', '--version', dest='version', default='ResUNet', type=str, help='version of models')
parser.add_argument('-t', '--train_mode', dest='mode', default='base', type=str, help='base, adv')
parser.add_argument('-e', '--epochs', dest='epochs', default=100, type=int, help='number of epochs')
parser.add_argument('-b', '--batch-size', dest='batchsize', default=16, type=int, help='batch size')
parser.add_argument('-m', '--model', metavar='FILE', help="pretrained model")
parser.add_argument('-l', '--learning-rate', dest='lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('-g', '--gpu', action='store_true', dest='gpu', default=True, help='use cuda')
parser.add_argument('-c', '--load', dest='load', default=False, help='load file model')
parser.add_argument('-r', '--ratio', dest='ratio', type=float, default=0, help='ratio of bce and dice')
args = parser.parse_args()


def cal_loss(predict, target, weight=0):
    bce = nn.BCELoss()
    # bce = FocalLoss()
    dice = dice_loss()
    loss = (1 - weight) * bce(predict, target) + weight * dice(predict, target)
    return loss


def eval_net(net, dataset, weight, gpu=True):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    tot = 0
    for i, b in enumerate(dataset):
        img = b[0]
        true_mask = b[1]

        if gpu:
            img = img.cuda()
            true_mask = true_mask.cuda()

        mask_pred = net(img)
        mask_pred = mask_pred.cpu()
        true_mask = true_mask.cpu()

        mask_pred = (mask_pred > 0.5).float()

        loss = cal_loss(mask_pred, true_mask, weight)

        tot += loss
    net.train()
    return tot.item() / (i + 1)


def create_logger(dir_checkpoint):
    if not os.path.exists(dir_checkpoint):
        os.mkdir(dir_checkpoint)

    logger = logging.getLogger(args.version)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s : %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # file saved in log/ path
    fh = logging.FileHandler(dir_checkpoint + args.version + '_' + timestamp + '.txt')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    # print in screen
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


'''
    Baseline Training
'''
def train_base(net, mode='mars', epochs=100, batch_size=16, lr=0.0001, save_cp=True, weight=0.5, gpu=True):
    dir_checkpoint = 'weights/' + args.version + '_' + timestamp + '/'

    dataset = Mars(root + '/data/', mode=mode, labeled=True, train_phase=True, aug=True)
    dataloader = data.DataLoader(dataset, batch_size, shuffle=True)

    test_set = Mars(root + '/data/', labeled=True, train_phase=False, aug=False)

    logger = create_logger(dir_checkpoint)
    logger.info(args)
    logger.info(net)

    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=0.0005)
    for epoch in range(epochs):
        logger.info('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        net.train()
        epoch_loss = 0
        batch_iterator = iter(dataloader)

        for i, b in enumerate(dataloader):
            try:
                imgs, true_masks = next(batch_iterator)
            except StopIteration:
                batch_iterator = iter(dataloader)
                imgs, true_masks = next(batch_iterator)

            if gpu:
                imgs = imgs.cuda()
                true_masks = true_masks.cuda()

            masks_pred = net(imgs)

            masks_probs_flat = masks_pred.view(-1)
            true_masks_flat = true_masks.view(-1)

            # using the combination of dice and bce
            loss = cal_loss(masks_probs_flat, true_masks_flat, weight=weight)

            epoch_loss += loss.item()

            logger.info('Epoch:{}/{}, {} ---lr: {:.5f} --- loss: {:.6f}'.
                        format(str(epoch + 1).zfill(2), epochs, str(i + len(dataloader) * epoch).zfill(5), lr,
                               loss.item()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        logger.info('Epoch Loss: {}'.format(epoch_loss / i))

        if True:
            val_loss = 0
            p, r, d, jard = predict(net, test_set, gpu=True)
            logger.info('Epoch:{}/{}, Train loss:{}, Val loss:{}, Precision: {}, Recall: {}, Dice: {}, Jard:{}\n'.
                        format(epoch + 1, epochs, epoch_loss / i, val_loss, p, r, d, jard))

        if save_cp:
            torch.save(net.state_dict(), dir_checkpoint + '{}.pth'.format(str(epoch + 1).zfill(3)))
            print('Checkpoint {} saved !'.format(epoch + 1))


'''
    Adversarial Training
'''


def train_adv(net,  epochs=100, batch_size=16, lr=0.0001, save_cp=True, weight=0.5, gpu=True):
    dis = Dis(1).cuda()

    dir_checkpoint = 'weights/' + args.version + '_' + timestamp + '/'

    dataset = Mars(root + '/data/', mars=False, labeled=True, train_phase=True)
    dataloader = data.DataLoader(dataset, batch_size, shuffle=True)

    test_set = Mars(root + '/data/', mars=True, labeled=True, train_phase=False, aug=False)

    logger = create_logger(dir_checkpoint)
    logger.info(args)
    logger.info(net)

    optimG = optim.Adam(net.parameters(), lr=lr, weight_decay=0.0005)
    optimD = optim.Adam(dis.parameters(), lr=lr, weight_decay=0.0005)
    bce = nn.BCELoss()

    for epoch in range(epochs):
        logger.info('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        net.train()
        epoch_loss = 0
        batch_iterator = iter(dataloader)

        for i, b in enumerate(dataloader):
            try:
                imgs, true_masks = next(batch_iterator)
            except StopIteration:
                batch_iterator = iter(dataloader)
                imgs, true_masks = next(batch_iterator)

            if gpu:
                imgs = imgs.cuda()
                true_masks = true_masks.cuda()

            with torch.no_grad():
                cpmap = net(Variable(imgs))

            N = cpmap.size()[0]
            H = cpmap.size()[2]
            W = cpmap.size()[3]

            # Generate the Real and Fake Labels
            targetf = Variable(torch.zeros((N, H, W)).long(), requires_grad=False)
            targetr = Variable(torch.ones((N, H, W)).long(), requires_grad=False)
            if gpu:
                targetf = targetf.float().cuda()
                targetr = targetr.float().cuda()

            ##########################
            # DISCRIMINATOR TRAINING #
            ##########################
            optimD.zero_grad()

            # Train on Real
            confr = dis(true_masks)

            LDr = (1 - 0.1) * bce(confr.view(-1), targetr.view(-1))
            LDr = LDr + 0.1 * bce(confr.view(-1), targetf.view(-1))

            LDr.backward()

            # Train on Fake
            conff = dis(cpmap)
            LDf = bce(conff.view(-1), targetf.view(-1))
            LDf.backward()

            optimD.step()

            ######################
            # GENERATOR TRAINING #
            #####################
            optimG.zero_grad()
            optimD.zero_grad()

            cmap = net(imgs)
            conff = dis(cpmap)

            LGce = bce(cmap.view(-1), true_masks.view(-1))
            LGadv = bce(conff.view(-1), targetr.view(-1))
            LGseg = LGce + 0.5 * LGadv

            LGseg.backward()
            optimG.step()
            optimD.step()

            logger.info("[{}][{}] LD: {:.4f} LDfake: {:.4f} LD_real: {:.4f} LG: {:.4f} LG_ce: {:.4f} LG_adv: {:.4f}"  \
                    .format(epoch, i, (LDr + LDf).item(), LDr.item(), LDf.item(), LGseg.item(), LGce.item(), LGadv.item()))

        logger.info('Epoch Loss: {}'.format(epoch_loss / i))

        if 1:
            val_loss = 0
            p, r, d, jard = predict(net, test_set, gpu=True)
            logger.info('Epoch:{}/{}, Train loss:{}, Val loss:{}, Precision: {}, Recall: {}, Dice: {}, Jard:{}\n'.
                        format(epoch + 1, epochs, epoch_loss / i, val_loss, p, r, d, jard))

        if save_cp:
            torch.save(net.state_dict(), dir_checkpoint + '{}.pth'.format(str(epoch + 1).zfill(3)))
            print('Checkpoint {} saved !'.format(epoch + 1))

'''
    AutoEncoder Training
'''
def train_AE(net, epochs=100, batch_size=16, lr=0.0001, save_cp=True, weight=0.5, gpu=True):
    dir_checkpoint = 'weights/' + args.version + '_' + timestamp + '/'

    dataset = Mars(root + '/data/', mode='mars', labeled=False, train_phase=True)
    dataloader = data.DataLoader(dataset, batch_size, shuffle=True)

    logger = create_logger(dir_checkpoint)
    logger.info(args)
    logger.info(net)

    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=0.0005)
    for epoch in range(epochs):
        logger.info('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        net.train()
        epoch_loss = 0
        batch_iterator = iter(dataloader)

        for i, b in enumerate(dataloader):
            try:
                imgs, true_masks = next(batch_iterator)
            except StopIteration:
                batch_iterator = iter(dataloader)
                imgs, true_masks = next(batch_iterator)

            if gpu:
                imgs = imgs.cuda()

            masks_pred = net(imgs)

            masks_probs_flat = masks_pred.view(-1)
            true_masks_flat = imgs.view(-1)

            # using the combination of dice and bce
            loss = cal_loss(masks_probs_flat, true_masks_flat)

            epoch_loss += loss.item()

            logger.info('Epoch:{}/{}, {} ---lr: {:.5f} --- loss: {:.6f}'.
                        format(str(epoch + 1).zfill(2), epochs, str(i + len(dataloader) * epoch).zfill(5), lr,
                               loss.item()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        logger.info('Epoch Loss: {}'.format(epoch_loss / i))

        if save_cp:
            torch.save(net.state_dict(), dir_checkpoint + '{}.pth'.format(str(epoch + 1).zfill(3)))
            print('Checkpoint {} saved !'.format(epoch + 1))


if __name__ == '__main__':
    if args.version == 'UNet':
        model = UNet(n_channels=3, n_classes=1)
    elif args.version == 'ResUNet':
        model = Res_UNet(1)

    if args.load:
        l = torch.load(args.load)
        l['final.weight'] = l['final.weight'][:1, :, :, :]
        l['final.bias'] = l['final.bias'][:1]
        model.load_state_dict(l)
        print('Model loaded from {}'.format(args.load))

    if args.gpu:
        # torch.backends.cudnn.benchmark = True # faster convolutions, but more memory
        model = model.cuda()
        # net = nn.DataParallel(net)
        if args.model:
            model.load_state_dict(torch.load(args.model))

    try:
        if args.mode == 'base':
            train_base(net=model, epochs=args.epochs, batch_size=args.batchsize, lr=args.lr, weight=args.ratio)
        elif args.mode == 'earth':
            train_base(net=model, mode='earth', epochs=args.epochs, batch_size=args.batchsize, lr=args.lr, weight=args.ratio)
        elif args.mode == 'earth_mars':
            train_base(net=model, mode='earth_mars', epochs=args.epochs, batch_size=args.batchsize, lr=args.lr, weight=args.ratio)
        elif args.mode == 'adv':
            train_adv(net=model, epochs=args.epochs, batch_size=args.batchsize, lr=args.lr, weight=args.ratio)
        elif args.mode == 'ae':
            train_AE(net=model, epochs=args.epochs, batch_size=args.batchsize, lr=args.lr, weight=args.ratio)
    except KeyboardInterrupt:
        torch.save(model.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
