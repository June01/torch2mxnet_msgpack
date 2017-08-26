#! /usr/bin/env python
#--------------------------------------------------
# Load mxnet model parameters from message pack for context-encoder gan
#
# Written by June Xie
# Date: 05/24/2017
# Copyright (c) 2017
#--------------------------------------------------
import os
import numpy as np
import umsgpack as mp
import mxnet as mx

def load_conv(save_path, name):
    """Load convolution layer parameters"""
    fname = os.path.join(save_path, '%s.msg' % name)
    with open(fname) as f:
        weight = mp.unpack(f)
        bias   = mp.unpack(f)

        weight = np.asarray(weight).astype(np.float32)
        bias   = np.asarray(bias).astype(np.float32)

        weight = mx.nd.array(weight)
        bias   = mx.nd.array(bias)
        print('load_conv done %s ', name)
        return weight, bias

def load_fullconv(save_path, name):
    """Load convolution layer parameters"""
    fname = os.path.join(save_path, '%s.msg' % name)
    with open(fname) as f:
        weight = mp.unpack(f)
        bias   = mp.unpack(f)

        weight = np.asarray(weight).astype(np.float32)
        bias   = np.asarray(bias).astype(np.float32)

        weight = mx.nd.array(weight)
        bias   = mx.nd.array(bias)
        print('load_fullconv done %s ', name)
        return weight, bias

def load_batch_norm(save_path, name):
    """Load batch normalization layer parameters"""
    fname = os.path.join(save_path, '%s.msg' % name)
    with open(fname) as f:
        gamma = mp.unpack(f)
        beta  = mp.unpack(f)
        mean  = mp.unpack(f)
        var   = mp.unpack(f)

        gamma = np.asarray(gamma).astype(np.float32)
        beta  = np.asarray(beta).astype(np.float32)
        mean  = np.asarray(mean).astype(np.float32)
        var   = np.asarray(var).astype(np.float32)

        gamma  = mx.nd.array(gamma)
        beta   = mx.nd.array(beta)
        mean   = mx.nd.array(mean)
        var    = mx.nd.array(var)
        print('load_batch_norm %s ', name)
        return gamma, beta, mean, var

def import_params(save_path):
    """Import parameters from msgpack"""
    if not os.path.exists(save_path):
        raise Exception('Path of parameters not exists: %s' % save_path)

    arg_params = {}
    aux_params = {}

    # Convolution layers parameters
    names = ['l1_conv', 'l3_conv', 'l6_conv', 'l9_conv', 'l12_conv', 'l15_conv']
    for name in names:
        mx_w, mx_b = load_conv(save_path, name)
        arg_params['%s_weight' % name] = mx_w
        arg_params['%s_bias' % name] = mx_b

    # Batch normalization layers parameters
    names = ['l4_bn', 'l7_bn', 'l10_bn', 'l13_bn', 'l16_bn', 'l19_bn', 'l22_bn', 'l25_bn', 'l28_bn']
    for name in names:
        gamma, beta, mean, var = load_batch_norm(save_path, name)
        arg_params['%s_gamma' % name] = gamma
        arg_params['%s_beta' % name]  = beta
        aux_params['%s_moving_mean' % name] = mean
        aux_params['%s_moving_var' % name]  = var

    # FullConvolution layers parameters
    names = ['l18_conv', 'l21_conv', 'l24_conv', 'l27_conv', 'l30_conv']
    for name in names:
        mx_w, mx_b = load_conv(save_path, name)
        arg_params['%s_weight' % name] = mx_w
        arg_params['%s_bias' % name] = mx_b

    return arg_params, aux_params

# if __name__ == '__main__':

#     save_path = './msgpack/'

#     arg_params, aux_params = import_params(save_path)

#     print('Arguments:')
#     for p in sorted(arg_params):
#         print(p, arg_params[p].shape)

#     print('Auxiliary states:')
#     for p in sorted(aux_params):
#         print(p, aux_params[p].shape)


#     print('Successfully imported %d argument parameters and %d auxiliary states' % (len(arg_params), len(aux_params)) )
