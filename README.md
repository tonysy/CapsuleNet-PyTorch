# CapsNet-PyTorch

I hvae crated `CapsuleConv` for PrimaryCapsule, `CapsuleLinear` for DigitCapsule

*minist_toy_example.py*: minist example from pytorch github code repo

# Usage

## Step-1: Set your environment
```
virtualenv -p python3 --no-site-packages .capsule_env

source .capsule_env/bin/activate
pip3 install http://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp35-cp35m-manylinux1_x86_64.whl 
pip3 install torchvision
```
## Step-2: Run

```
CUDA_VISIBLE_DEVICES=2 python main.py
```

# Todo

1. Test for cifar10
2. Re-Implement EM-Routing

# Other Implementations
- Kaggle (this version as self-contained notebook):
  - [MNIST Dataset](https://www.kaggle.com/kmader/capsulenet-on-mnist) running on the standard MNIST and predicting for test data
  - [MNIST Fashion](https://www.kaggle.com/kmader/capsulenet-on-fashion-mnist) running on the more challenging Fashion images.
- TensorFlow:
  - [naturomics/CapsNet-Tensorflow](https://github.com/naturomics/CapsNet-Tensorflow.git)   
  Very good implementation. I referred to this repository in my code.
  - [InnerPeace-Wu/CapsNet-tensorflow](https://github.com/InnerPeace-Wu/CapsNet-tensorflow)   
  I referred to the use of tf.scan when optimizing my CapsuleLayer.
  - [LaoDar/tf_CapsNet_simple](https://github.com/LaoDar/tf_CapsNet_simple)

- PyTorch:
  - [nishnik/CapsNet-PyTorch](https://github.com/nishnik/CapsNet-PyTorch.git)
  - [timomernick/pytorch-capsule](https://github.com/timomernick/pytorch-capsule)
  - [gram-ai/capsule-networks](https://github.com/gram-ai/capsule-networks)
  - [andreaazzini/capsnet.pytorch](https://github.com/andreaazzini/capsnet.pytorch.git)
  - [leftthomas/CapsNet](https://github.com/leftthomas/CapsNet)
  
- MXNet:
  - [AaronLeong/CapsNet_Mxnet](https://github.com/AaronLeong/CapsNet_Mxnet)
  
- Lasagne (Theano):
  - [DeniskaMazur/CapsNet-Lasagne](https://github.com/DeniskaMazur/CapsNet-Lasagne)

- Chainer:
  - [soskek/dynamic_routing_between_capsules](https://github.com/soskek/dynamic_routing_between_capsules)
