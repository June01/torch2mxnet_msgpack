-- --------------------------------------------------
-- Convert context-encoder-gan torch model into message pack
--
--  Written by June Xie
--  Date: 05/24/17
--  Copyright (c) 2017
-- --------------------------------------------------
require 'torch'
require 'nn'
-- require 'cudnn'
-- require 'bnn'
require 'paths'

mp = require 'MessagePack'
mp.set_array'without_hole'

-- torch.setdefaulttensortype it is a good answer
torch.setdefaulttensortype('torch.FloatTensor')

-- save convolution layer parameters
local save_conv = function(layer, name, save_path)
    print('save_conv'..name..'start')
    th_weight = layer.weight:float()
    th_bias = layer.bias:float()
    weight = {}
    for i = 1, th_weight:size(1) do
        weight[i] = {}
        for j = 1, th_weight:size(2) do
            weight[i][j] = {}
            for k = 1, th_weight:size(3) do
                weight[i][j][k] = {}
                for l = 1, th_weight:size(4) do
                    weight[i][j][k][l] = th_weight[i][j][k][l]
                end
            end
        end
    end

	bias = {}
	for i = 1 , th_bias:size(1) do
		bias[i] = th_bias[i]
	end

	mp_w = mp.pack(weight)
	mp_b = mp.pack(bias)

    file = io.open(save_path..name..".msg" , "w")
	file:write(mp_w)
	file:write(mp_b)
	file:close()
    print('save_conv'..name..'done')
end

-- save batch normalization layer parameters
local save_batch_norm = function(layer, name, save_path)
    print('save_batch_norm'..name..'start')
    th_weight = layer.weight:float() -- gamma
    th_bias   = layer.bias:float() -- beta
    th_mean   = layer.running_mean:float()
    th_var    = layer.running_var:float()
    
    gamma = {}
    beta  = {}
    mean  = {}
    var   = {}
    
    for i = 1, th_weight:size(1) do
        gamma[i] = th_weight[i]
        beta[i]  = th_bias[i]
        mean[i]  = th_mean[i]
        var[i]   = th_var[i]
    end

    mp_w = mp.pack(gamma)
    mp_b = mp.pack(beta)
    mp_m = mp.pack(mean)
    mp_v = mp.pack(var)

    file = io.open(save_path..name..".msg", "w")
	file:write(mp_w)
	file:write(mp_b)
	file:write(mp_m)
	file:write(mp_v)
	file:close()
    print('save_conv'..name..'done')
end

-- relu, leakyrelu and tanh donnot need parameters, only slope=0.2 in leakyrelu

-- save full convolution layer parameters
local save_fullconv = function(layer, name, save_path)
    print('save_fullconv'..name..'start')
    th_weight = layer.weight:float()
    th_bias = layer.bias:float()
    weight = {}
    for i = 1, th_weight:size(1) do
        weight[i] = {}
        for j = 1, th_weight:size(2) do
            weight[i][j] = {}
            for k = 1, th_weight:size(3) do
                weight[i][j][k] = {}
                for l = 1, th_weight:size(4) do
                    weight[i][j][k][l] = th_weight[i][j][k][l]
                end
            end
        end
    end

    bias = {}
    for i = 1 , th_bias:size(1) do
        bias[i] = th_bias[i]
    end

    mp_w = mp.pack(weight)
    mp_b = mp.pack(bias)

    file = io.open(save_path..name..".msg" , "w")
    file:write(mp_w)
    file:write(mp_b)
    file:close()
    print('save_conv'..name..'done')
end

-- export models
local export_model = function(model_path, save_path)
    local model = {}

    model = torch.load(model_path)

    os.execute('mkdir -p ' .. save_path)
    model:evaluate()

    print(model)
    save_conv(model:get(1):get(1), "l1_conv", save_path)
    save_conv(model:get(1):get(3), "l3_conv", save_path)
    save_conv(model:get(1):get(6), "l6_conv", save_path)
    save_conv(model:get(1):get(9), "l9_conv", save_path)
    save_conv(model:get(1):get(12), "l12_conv", save_path)
    save_conv(model:get(1):get(15), "l15_conv", save_path)

    save_batch_norm(model:get(1):get(4), "l4_bn", save_path)
    save_batch_norm(model:get(1):get(7), "l7_bn", save_path)
    save_batch_norm(model:get(1):get(10), "l10_bn", save_path)
    save_batch_norm(model:get(1):get(13), "l13_bn", save_path)
    save_batch_norm(model:get(2), "l16_bn", save_path)
    save_batch_norm(model:get(5), "l19_bn", save_path)
    save_batch_norm(model:get(8), "l22_bn", save_path)
    save_batch_norm(model:get(11), "l25_bn", save_path)
    save_batch_norm(model:get(14), "l28_bn", save_path)

    save_fullconv(model:get(4), "l18_conv", save_path)
    save_fullconv(model:get(7), "l21_conv", save_path)
    save_fullconv(model:get(10), "l24_conv", save_path)
    save_fullconv(model:get(13), "l27_conv", save_path)
    save_fullconv(model:get(16), "l30_conv", save_path)

    print('parameters are saved in: '..save_path)
end

local model_path = './models/imagenet_inpaintCenter.t7'
local save_path = './msgpack/'
if paths.filep(model_path) then
    export_model(model_path, save_path)
else
    print('Modwl not exists')
end
