import mxnet as mx
import numpy as np
import cv2
from torch.utils.serialization import load_lua
from PIL import Image
from torchvision.transforms import ToTensor
from collections import namedtuple
Batch = namedtuple('Batch', ['data'])

# load the data
sym, arg_params, aux_params = mx.model.load_checkpoint('./models/ip_mxnet', 0)
mod = mx.mod.Module(symbol = sym, context = mx.cpu())
mod.bind(for_training = False, data_shapes = [('data', (1, 3, 128,128))])
mod.set_params(arg_params, aux_params)

# load the image
# Attention:
# There are three difference between image.load and cv2.load:
# 1. cv2-BGR; image.load-RGB
# 2. cv2-300x400x3; image.load-3x300x400
# 3. cv2-255x255x3; image.load-3x(0-1)x(0-1)
# 4. cv2 index from 0; image.load index from 1

im = cv2.imread('./images/001_im.png')
# print(im[0,:,:])
im = cv2.resize(im,(128,128))
imf = im.astype(np.float)
imf = imf/255

imb = np.zeros([3,128,128])

imb[0,:,:] = imf[:,:,2]
imb[1,:,:] = imf[:,:,1]
imb[2,:,:] = imf[:,:,0]

imb = imb*2-1
# imb[:,36:92,36:92] = 0

# make sure the input is the same, especially the blank hole
# Ateention: float and integer in python
imb[0,36:92,36:92] = 2*117.0/255.0 - 1.0
imb[1,36:92,36:92] = 2*104.0/255.0 - 1.0
imb[2,36:92,36:92] = 2*123.0/255.0 - 1.0
# print(imb[0,:,:])
# print('------------------------------------')
# print(imb[1,:,:])
# print('------------------------------------')
# print(imb[2,:,:])
print(imb[0,35,35])
print(imb[0,36,36])
print(imb[0,37,37])

print(imb[0,91,91])
print(imb[0,92,92])
print(imb[0,93,93])

input = imb[np.newaxis, :]
input = Batch([mx.nd.array(input)])
# print(imb)

# forward the model to complement the image
mod.forward(input)
output = mod.get_outputs()[0].asnumpy()

out = np.zeros([64,64,3])

# convert the torch image format to numpy opencv and save
out[:,:,0] = output[0,2,:,:]
out[:,:,1] = output[0,1,:,:]
out[:,:,2] = output[0,0,:,:]
print(out)
# print(out)
out = (out+1)*0.5
out = out*255


cv2.imwrite('test.png', out)

'''
# output the internal results
# internals = mod.symbol.get_internals()
# # print(internals.list_outputs())
# output15_symbol = internals['l31_output']
# module15 = mx.model.FeedForward(ctx=mx.cpu(), symbol=output15_symbol, numpy_batch_size=1,
# 	                            arg_params=arg_params, aux_params=aux_params, allow_extra_params=True)
# module15_output = module15.predict(input)
'''

# compare the results between two array
# result11_py_np = input
# result11_py_np = tmp
# result11_th = load_lua('../torch2mxnet/t7_output.dat')
# result11_th_np = result11_th.numpy()

# print(max(result11_py_np.flatten()-result11_th_np.flatten()))
# print(min(result11_py_np.flatten()-result11_th_np.flatten()))
# print(result11_py_np.flatten())
# print(result11_th_np.flatten())
# diff = np.linalg.norm(result11_py_np.flatten()-result11_th_np.flatten())
# print(diff)

