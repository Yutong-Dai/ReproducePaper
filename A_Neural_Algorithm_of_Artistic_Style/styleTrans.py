'''
File: main.py
Author: Yutong Dai (yutongdai95@gmail.com)
File Created: 2021-03-21 22:19
Last Modified: 2021-03-21 23:17
--------------------------------------------
Description:
'''
import torch
import torchvision.models as models
import utils
import cv2
import torch.optim


def style_transfer(c='./db/pikachu.jpg', s='./db/pikachu.jpg', epochs=400,
                   c_layer=6, alpha=1, beta=6, layer_weights=0.5,
                   optimizer='adam', lr=0.1, scheduler='yes'):
    # load model
    model = models.vgg19(pretrained=True)
    model = model.cuda()
    # load image
    # contentImage
    img = cv2.imread(c)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (1024, 768))
    contentImage = torch.tensor(img / 255.0).float().cuda()
    # style image
    img = cv2.imread(s)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (1024, 768))
    styleImage = torch.tensor(img / 255.0).float().cuda()
    layers = utils.get_layers(model)

    aCs = utils.get_feature_maps(contentImage, layers, keep_grad=False)
    aSs = utils.get_feature_maps(styleImage, layers, keep_grad=False)

    torch.manual_seed(0)
    G = torch.rand(contentImage.shape, requires_grad=True, device="cuda")
    style_layer_weights = [layer_weights for i in range(16)]
    if optimizer == 'adam':
        optimizer = optim.Adam([G], lr)
    scheduler = None
    if scheduler != 'none':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    # learn stlye + contents

    for _ in range(epochs):
        optimizer.zero_grad()
        imgInput = torch.transpose(G.unsqueeze(0), 1, 3)
        aGs = utils.get_feature_maps(imgInput, layers)
        loss = utils.compute_total_cost(aGs, aCs, aSs, style_layer_weights,
                                        content_layer_idx=c_layer, alpha=alpha, beta=beta)
        loss.backward()
        optimizer.step()
        scheduler.step()
    return G.cpu()
