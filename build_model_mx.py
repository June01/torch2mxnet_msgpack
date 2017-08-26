import numpy as np
import mxnet as mx
from import_msgpack import import_params

# define encoder model
def encoder(data):
    conv11 = mx.symbol.Convolution(name = 'l1_conv', data = data, kernel = (4, 4), stride = (2, 2), pad = (1, 1), num_filter = 64)
    relu11 = mx.symbol.LeakyReLU(name = 'l2', data = conv11, slope = 0.2, act_type = 'leaky')

    conv12 = mx.symbol.Convolution(name = 'l3_conv', data = relu11, kernel = (4, 4), stride = (2, 2), pad = (1, 1), num_filter = 64)
    bn12 = mx.symbol.BatchNorm(name='l4_bn', data = conv12, eps = 1e-05, momentum = 0.1, fix_gamma = False)
    relu12 = mx.symbol.LeakyReLU(name = 'l5', data = bn12, slope = 0.2, act_type = 'leaky')

    conv13 = mx.symbol.Convolution(name = 'l6_conv', data = relu12, kernel = (4, 4), stride = (2, 2), pad = (1, 1), num_filter = 128)
    bn13 = mx.symbol.BatchNorm(name='l7_bn', data = conv13, eps = 1e-05, momentum = 0.1, fix_gamma = False)
    relu13 = mx.symbol.LeakyReLU(name = 'l8', data = bn13, slope = 0.2, act_type = 'leaky')

    conv14 = mx.symbol.Convolution(name = 'l9_conv', data = relu13, kernel = (4, 4), stride = (2, 2), pad = (1, 1), num_filter = 256)
    bn14 = mx.symbol.BatchNorm(name='l10_bn', data = conv14, eps = 1e-05, momentum = 0.1, fix_gamma = False)
    relu14 = mx.symbol.LeakyReLU(name = 'l11', data = bn14, slope = 0.2, act_type = 'leaky')

    conv15 = mx.symbol.Convolution(name = 'l12_conv', data = relu14, kernel = (4, 4), stride = (2, 2), pad = (1, 1), num_filter =512)
    bn15 = mx.symbol.BatchNorm(name='l13_bn', data = conv15, eps = 1e-05, momentum = 0.1, fix_gamma = False)
    relu15 = mx.symbol.LeakyReLU(name = 'l14', data = bn15, slope = 0.2, act_type = 'leaky')

    conv16 = mx.symbol.Convolution(name = 'l15_conv', data = relu15, kernel = (4, 4), stride = (1, 1), pad = (0, 0), num_filter =4000)

    return conv16

#define inpainting model
def inpaintModel(data):

    data = encoder(data)
    bn21 = mx.symbol.BatchNorm(name='l16_bn', data = data, eps = 1e-05, momentum = 0.1, fix_gamma = False)
    relu21 = mx.symbol.LeakyReLU(name = 'l17', data = bn21, slope = 0.2, act_type = 'leaky')
    convt21 = mx.symbol.Deconvolution(name = 'l18_conv', data = relu21, kernel = (4, 4), stride = (1, 1), pad = (0, 0), num_filter = 512)

    bn22 = mx.symbol.BatchNorm(name='l19_bn', data = convt21, eps = 1e-05, momentum = 0.1, fix_gamma = False)
    relu22 = mx.symbol.Activation(name = 'l20', data = bn22, act_type = 'relu')
    convt22 = mx.symbol.Deconvolution(name = 'l21_conv', data = relu22, kernel = (4, 4), stride = (2, 2), pad = (1, 1), num_filter = 256)

    bn23 = mx.symbol.BatchNorm(name='l22_bn', data = convt22, eps = 1e-05, momentum = 0.1, fix_gamma = False)
    relu23 = mx.symbol.Activation(name = 'l23', data = bn23, act_type = 'relu')
    convt23 = mx.symbol.Deconvolution(name = 'l24_conv', data = relu23, kernel = (4, 4), stride = (2, 2), pad = (1, 1), num_filter = 128)

    bn24 = mx.symbol.BatchNorm(name='l25_bn', data = convt23, eps = 1e-05, momentum = 0.1, fix_gamma = False)
    relu24 = mx.symbol.Activation(name = 'l26', data = bn24, act_type = 'relu')
    convt24 = mx.symbol.Deconvolution(name = 'l27_conv', data = relu24, kernel = (4, 4), stride = (2, 2), pad = (1, 1), num_filter = 64)

    bn25 = mx.symbol.BatchNorm(name='l28_bn', data = convt24, eps = 1e-05, momentum = 0.1, fix_gamma = False)
    relu25 = mx.symbol.Activation(name = 'l29', data = bn25, act_type = 'relu')
    convt25 = mx.symbol.Deconvolution(name = 'l30_conv', data = relu25, kernel = (4, 4), stride = (2, 2), pad = (1, 1), num_filter = 3)

    tanh21 = mx.symbol.Activation(name = 'l31', data = convt25, act_type = 'tanh')

    return tanh21

save_path = './msgpack/'

arg_params, aux_params = import_params(save_path)
print('Successfully imported %d argument parameters and %d auxiliary states' % (len(arg_params), len(aux_params)) )

print(arg_params)
print(aux_params)
data = mx.symbol.Variable('data')
sym = inpaintModel(data)
mod_mx = mx.mod.Module(symbol = sym, context = mx.cpu())
mod_mx.bind(for_training = False, data_shapes = [('data', (1, 3, 128, 128))])
mod_mx.set_params(arg_params, aux_params)
mod_mx.save_checkpoint('ip_mxnet', 0)

print 'ok'

