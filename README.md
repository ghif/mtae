# Multi-task Autoencoders for Domain Generalization
==========

This code contains the Python implementation of the Multitask Autoencoder (MTAE) algorithm based on the following paper:

M. Ghifary, B. Kleijn, M. Zhang, D. Balduzzi
**Domain Generalization for Object Recognition with Multi-task Autoencoders**
accepted in International Conference on Computer Vision (**ICCV 2015**), Santiago, Chile
[http://arxiv.org/abs/1508.07680](http://arxiv.org/abs/1508.07680)

Please cite the above paper when using this code.

_Notes_:
- the implementation is based on the theano wrapper called [keras.io](keras.io)
- still not well commented
- only works for the provided MNIST dataset (with 6 rotated views)

For questions and bug reports, please send me an email at _mghifary[at]gmail.com_.

### Prerequisites
1. The following frameworks/libraries must be installed:
	- Python (version 2.7 or higher)
	- Numpy (e.g. `pip install numpy`)
	- [http://deeplearning.net/software/theano/](Theano)
	- [keras.io](Keras)

2. Clone this repository
```sh
git clone https://github.com/ghif/mtae.git
```