'''
File: main.py
Author: Yutong Dai (yutongdai95@gmail.com)
File Created: 2021-03-21 22:19
Last Modified: 2021-03-24 18:19
--------------------------------------------
Description:
'''
import os
import torch
import torchvision.models as models
import utils
import cv2
import torch.optim as optim
from torchvision.utils import save_image
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--c", help="Path of the content image", type=str)
parser.add_argument("--s", help="Path of the style image", type=str)
parser.add_argument("--savename", help="working folder save temp results", type=str, default='temp')
parser.add_argument("--G_pretrained", help="Path of the pre-trained image", type=str, default='None')
parser.add_argument("--epochs", help="Number of epochs to run, default=10000", type=int, default=10000)
parser.add_argument("--alpha", help="Weight for the style loss, default=10e6", type=int, default=1)
parser.add_argument("--beta", help="Weight for the content loss, default=10e-4", type=float, default=1e4)
parser.add_argument("--c_layer", help="Weight for the content loss, default=10e-4", type=int, default=5)
parser.add_argument("--printevery", help="Print the progress every K iretations, default=1000", type=int, default=500)
parser.add_argument("--starting", help="statring epoch (for continute training from checkpoint), default=0", type=int, default=0)


def style_transfer(c='./db/pikachu.jpg', s='./db/starry.jpg', savename='pikachu-starry',
                   G_pretrained=None, epochs=10000, c_layer=5, alpha=1, beta=1e4, printevery=500, starting=0):
    print(f"Content Image:{c} | Style Image:{s} | savename: {savename}", flush=True)
    # load model
    model = models.vgg19(pretrained=True)
    model = model.cuda()
    for param in model.parameters():
        param.requires_grad = False
    # load image
    # contentImage
    img = cv2.imread(c)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, c = img.shape
    contentImage = torch.tensor(img / 255.0).float().cuda()
    # style image
    img = cv2.imread(s)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (w, h))
    styleImage = torch.tensor(img / 255.0).float().cuda()
    layers = utils.get_layers(model)

    aCs = utils.get_feature_maps(contentImage, layers)
    aSs = utils.get_feature_maps(styleImage, layers)
    if G_pretrained != 'None':
        G = G_pretrained
    else:
        # torch.manual_seed(0)
        # G = torch.rand(contentImage.shape, requires_grad=True, device="cuda")
        G = contentImage.detach().clone().requires_grad_(True).cuda()
    style_layer_weights = [1.0 / 16 for i in range(16)]

    optimizer = optim.AdamW([G], 0.001)
    if not os.path.exists(f'./generated/{savename}'):
        os.mkdir(f'./generated/{savename}')
    # learn stlye + contents
    for it in range(starting, starting + epochs):
        optimizer.zero_grad()
        aGs = utils.get_feature_maps(G, layers)
        loss, content_cost, style_cost = utils.compute_total_cost(aGs, aCs, aSs, style_layer_weights,
                                                                  content_layer_idx=c_layer, alpha=alpha, beta=beta)
        if (it + 1) % printevery == 0 or it == 0:
            print(f'iters: {it+1:5d} | loss:{loss.data.cpu().item():2.3e} | content: {content_cost.item():2.3e} | style_cost:{style_cost.item():2.3e}', flush=True)
            save_image(G.permute(2, 0, 1).cpu().detach(), fp='./generated/{}/iter_{}.png'.format(savename, it+1))
        loss.backward()
        optimizer.step()


if __name__ == "__main__":

    # parse arguments
    args = parser.parse_args()
    c = args.c
    s = args.s
    savename = args.savename
    G_pretrained = args.G_pretrained
    epochs = args.epochs
    c_layer = args.c_layer
    alpha = args.alpha
    beta = args.beta
    printevery = args.printevery
    starting = args.starting
    # run the model
    style_transfer(c, s, savename, G_pretrained, epochs, c_layer, alpha, beta, printevery, starting)
