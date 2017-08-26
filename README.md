# torch2mxnet_msgpack
This is a method that can convert torch model to mxnet and the params are delivered by msgpack. Msgpack can be also used to other model convertion. The original model consists of many kind of layers, such as conv, relu, leakyRelu, deconv, batchNorm, tanh, and so on. So it almost includes most kinds of the convertion case.

To make your convertion runs smoothly, firstly, you should install the [torch](http://torch.ch/docs/getting-started.html), python and the corresponding msgpack package separately.

### Install MessagePack
- install [u-masgpack-python](https://github.com/vsergeev/u-msgpack-python)
```shell
sudo pip install u-msgpack-python
```

- install lua messagepack
```shell
luarocks install lua-messagepack
```
### Clone the repository
```shell
git clone https://github.com/June01/torch2mxnet_msgpack.git
```
### Export the parameters of torch model to messagepack format
```shell
cd torch2mxnet_msgpack
th export_msgpack.lua
```
Attention: If your model structure is not the same with me, please remember to modify it and aslo```model_path``` and ```save_path```.

### Export to build mxnet model
```shell
python build_model_mx.py
```
Attention: If your model structure is not the same with me, please remember to modify it and also ```import_msgpack.py```.

If you have questions with it, please open an issue or ｀｀｀email me｀｀｀(xietingting14@nudt.edu.cn).
