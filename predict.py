import argparse
import os

import numpy as np
import torch

from PIL import Image

from models import UNet,Res_UNet
# from utils import dense_crf
from data.mars import mean, std


def predict(net, data, gpu):
    net.eval()
    P, R, D, F1 = 0, 0, 0, 0
    for img, mask in data:
        if gpu:
            net = net.cuda()
            img = img.cuda()
        img = img.view(1, 3, 672, 672)
        mask_pred = net(img)

        mask = mask.view(672, 672)
        mask = mask.type(torch.ByteTensor)

        mask_pred = mask_pred.data
        mask_pred = mask_pred.squeeze(0).squeeze(0)
        mask_pred = mask_pred.cpu().numpy()

        # mask_out = dense_crf(img, mask_pred)
        mask_out = (mask_pred > 0.5).astype(np.float)
        # print(mask_out)
        mask_out = torch.from_numpy(mask_out).type(torch.ByteTensor)

        p, r, d, f1 = eval(mask, mask_out)
        P += p
        R += r
        D += d
        F1 += f1

    P = P/(len(data))
    R = R/(len(data))
    D = D/(len(data))
    F1 = F1/(len(data))
    # print('Evaluation Results:')
    # print('Precision: ', P)
    # print('Recall: ', R)
    # print('Dice: ', D)
    # print('Jaccard: ', F1)
    return P, R, D, F1


def eval(mask, result):
    intersection = float(torch.sum(mask & result))
    union = float(torch.sum(mask) + torch.sum(result))
    precision = intersection / (float(torch.sum(result)) + 1e-10)
    recall = intersection / (float(torch.sum(mask)) + 1e-10)
    dice = 2 * intersection / (union + 1e-10)
    jard = (intersection + 1e-10) / (union - intersection + 1e-10)
    return precision, recall, dice, jard


def predict_img(net, full_img, out_threshold=0.5, use_dense_crf=True, use_gpu=False):
    net.eval()
    img = (np.array(full_img, dtype=np.float32) - mean) / std
    img_ = np.transpose(img, axes=[2, 0, 1])
    img_ = torch.from_numpy(img_).unsqueeze_(0)
    # print(img_.shape)
    if use_gpu:
        img_ = img_.cuda()

    with torch.no_grad():
        output = net(img_)

        probs = output.squeeze(0)

        mask = probs.squeeze().cpu().numpy()

    # if use_dense_crf:
    #     mask = dense_crf(np.array(img).astype(np.uint8), mask)

    return mask > out_threshold
    # return mask


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--version', dest='version', default='UNet', type=str, help='version of models')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE', help="Specify the file in which is stored the model")
    parser.add_argument('--input', '-i', metavar='INPUT', default='test/imgs/', help='input file path', required=True)
    parser.add_argument('--output', '-o', metavar='INPUT', default='test/result/', help='output file path')
    parser.add_argument('--cpu', '-c', action='store_true', help="Do not use the cuda version of the net", default=False)
    parser.add_argument('--no-save', '-n', action='store_true', help="Do not save the output masks", default=False)
    parser.add_argument('--crf', '-r', type=bool, help="Use dense CRF postprocessing", default=False)
    parser.add_argument('--mask-threshold', '-t', type=float, help="Minimum probability value to consider a mask pixel white", default=0.5)

    return parser.parse_args()


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))


if __name__ == "__main__":
    args = get_args()
    print(args)
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    in_files = os.listdir(args.input)

    if args.version == 'UNet':
        net = UNet(n_channels=3, n_classes=1)
    elif args.version == 'ResUNet':
        net = Res_UNet(1)

    print("Loading model {}".format(args.model))

    if not args.cpu:
        print("Using CUDA version of the net, prepare your GPU !")
        net.cuda()
        # net = torch.nn.DataParallel(net)
        net.load_state_dict(torch.load(args.model))
    else:
        net.cpu()
        net.load_state_dict(torch.load(args.model, map_location='cpu'))
        print("Using CPU version of the net, this may be very slow")

    print("Model loaded !")

    for i, fn in enumerate(in_files):
        # print("\nPredicting image {} ...".format(fn))

        img = Image.open(args.input + '/' + fn)
        # img = img.resize((224, 224))

        mask = predict_img(net=net,
                           full_img=img,
                           out_threshold=args.mask_threshold,
                           use_dense_crf=args.crf,
                           use_gpu=not args.cpu)

        if not args.no_save:
            out_fn = args.output + '/' + fn.split('.')[0] + '.png'
            result = mask_to_image(mask)
            result.save(out_fn)
            print("Mask saved to {}".format(out_fn))
