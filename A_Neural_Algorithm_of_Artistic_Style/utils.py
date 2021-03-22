'''
File: utils.py
Author: Yutong Dai (yutongdai95@gmail.com)
File Created: 2021-03-21 22:17
Last Modified: 2021-03-21 23:35
--------------------------------------------
Description:
'''
import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
import torch
import numpy as np
import pprint


def get_layers(model):
    layers = []
    for idx, m in enumerate(model.features.modules()):
        if idx > 0:
            layers.append([m, isinstance(m, torch.nn.Conv2d)])
    return layers


def get_feature_maps(img, layers, keep_grad=True):
    """
        get outputs from conv2d layers
    """
    imgInput = torch.transpose(img.unsqueeze(0), 1, 3)
    aCs = []
    for l, isConv in layers:
        imgInput = l(imgInput)
        if isConv:
            if keep_grad:
                aCs.append(imgInput)
            else:
                aCs.append(imgInput.detach())
    return aCs


def compute_layer_content_cost(a_G, a_C):
    """
    Computes the content cost

    Arguments:
    a_C -- tensor of dimension (1, n_C, n_H, n_W), hidden layer activations representing content of the image C 
    a_G -- tensor of dimension (1, n_C, n_H, n_W), hidden layer activations representing content of the image G

    Returns: 
    J_content -- scalar that you compute using equation 1 above.
    """
    n_bacth, n_feature_maps, n_H, n_W = a_C.shape
    m = n_feature_maps
    n = n_H * n_W
    a_C_mat = a_C.view(n_bacth, n_feature_maps, -1)
    a_G_mat = a_G.view(n_bacth, n_feature_maps, -1)
    J_content = torch.sum((a_C_mat - a_G_mat) ** 2) / (4 * m * n)
    return J_content


def compute_content_cost(aGs, aCs):
    """
    Computes the overall style cost from several chosen layers

    Arguments:
    aGs, aCc -- activations after conv2d for G ans C images

    Returns: 
    J_content -- tensor representing a scalar value, style cost defined above
    """

    # initialize the overall style cost
    J_content = 0.0
    for i in range(len(aGs)):
        layer_cost = compute_layer_content_cost(aGs[i], aCs[i])
        J_content += layer_cost
    return J_content


def gram_matrix(A):
    """
    Argument:
    A -- matrix of shape (n_C, n_H*n_W)

    Returns:
    GA -- Gram matrix of A, of shape (n_C, n_C)
    """
    GA = torch.matmul(A, A.T)
    return GA


def compute_layer_style_cost(a_S, a_G):
    """
    Arguments:
    a_S -- tensor of dimension (1, n_C, n_H, n_W), hidden layer activations representing style of the image S 
    a_G -- tensor of dimension (1, n_C, n_H, n_W), hidden layer activations representing style of the image G

    Returns: 
    J_style_layer -- tensor representing a scalar value, style cost defined above by equation (2)
    """
    n_bacth, n_feature_maps, n_H, n_W = a_S.shape
    m = n_feature_maps
    n = n_H * n_W
    # dirty coding: n_bacth is expected to be 1
    a_S_mat = a_S.view(n_bacth, n_feature_maps, -1)[0]
    a_G_mat = a_G.view(n_bacth, n_feature_maps, -1)[0]
    gramS = gram_matrix(a_S_mat)
    gramG = gram_matrix(a_G_mat)
    J_style_layer = torch.sum((gramS - gramG)**2) / ((2 * m * n)**2)
    return J_style_layer


def compute_style_cost(aGs, aSc, STYLE_LAYERS_WEIGHTS):
    """
    Computes the overall style cost from several chosen layers

    Arguments:
    aGs, aSc -- activations after conv2d for G ans S images
    STYLE_LAYERS -- A python list containing a coefficient for each cost layer

    Returns: 
    J_style -- tensor representing a scalar value, style cost defined above
    """

    # initialize the overall style cost
    J_style = 0.0
    for i in range(len(STYLE_LAYERS_WEIGHTS)):
        layer_cost = compute_layer_style_cost(aSc[i], aGs[i])
        J_style += STYLE_LAYERS_WEIGHTS[i] * layer_cost
    return J_style


def compute_total_cost(aGs, aCs, aSs, style_layer_weights,
                       content_layer_idx, alpha, beta):
    """
    Computes the style cost + content cost

    Arguments:
    aGs, aCs, aSs -- activations after conv2d for G, C, S images
    STYLE_LAYERS -- A python list containing a coefficient for each cost layer
    alpha, beta -- weights

    Returns: 
    total -- tensor representing a scalar value, cost defined above
    """
    content_cost = compute_layer_content_cost(aCs[content_layer_idx], aGs[content_layer_idx])
    style_cost = compute_style_cost(aGs, aSs, style_layer_weights)
    total = alpha * content_cost + beta * style_cost
    return total
