# Multi-task Autoencoders for Domain Generalization
==========

This code contains the Python implementation of the Multitask Autoencoder (MTAE) algorithm based on the following paper:

M. Ghifary, B. Kleijn, M. Zhang, D. Balduzzi.<br/>
**Domain Generalization for Object Recognition with Multi-task Autoencoders**,<br/>
accepted in International Conference on Computer Vision (**ICCV 2015**), Santiago, Chile.<br/>
[[pre-print](http://arxiv.org/abs/1508.07680)]

Please cite the above paper when using this code.

_Notes_:
- This version is based on the Theano wrapper called [keras.io](keras.io) and was written after the paper finished. The prior code of this work was implemented in MATLAB. 
- Currently only works for the provided MNIST dataset (with 6 rotated views)
- Still not well commented

_TO DO_:
- RAND-SEL procedure
- SVM classification
- Supervised Finetuning


For questions and bug reports, please send me an email at _mghifary[at]gmail.com_.

### Prerequisites

1. The following frameworks/libraries must be installed:
  1. Python (version 2.7 or higher)
  2. Numpy (e.g. `pip install numpy`)
  3. [Theano](http://deeplearning.net/software/theano/)
  4. [Keras](keras.io)
2. Clone this repository, e.g.: ``` git clone https://github.com/ghif/mtae.git```
3. Run the main program to reproduce either Figure 4(c) or (d): ``` ./run_mtae_gpu.sh```
	* if you have a GPU, make sure that the nvcc compiler path is included in the environment variables.
