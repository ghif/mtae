
import numpy as np
import sys
import gzip
import cPickle as pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib

def load_rotated_mnist(datapath='MNIST_6rot.pkl.gz',left_out_idx=0):
	domains = pickle.load(gzip.open(datapath,'rb'))

	src_domains = domains[:] # clone the list
	del src_domains[left_out_idx]

	(X_test, y_test) = domains[left_out_idx]
	y_test = domains[left_out_idx][1]

	return src_domains, (X_test, y_test)

def get_corrupted_output(X, corruption_level=0.3):
    return np.random.binomial(1, 1-corruption_level, X.shape) * X

def show_images(X, padsize=1, padval=0, grayscale=False, filename=None, conv=False):
    data = np.copy(X)
    if conv:
        [n, c, d1, d2] = data.shape
        if c == 1:
            data = data.reshape((n, d1, d2))
        else:
            data = data.transpose(0,2,3,1)
    else:
        # print(data.shape)
        [n, d] = data.shape
        s = int(np.sqrt(d))
        data = data.reshape((n, s, s))

    vis_square(data, padsize=padsize, padval=padval, grayscale=grayscale, filename=filename)

# take an array of shape (n, height, width) or (n, height, width, channels)
# and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
def vis_square(data, padsize=1, padval=0, grayscale=False, filename=None):

    # print('min : ', np.min(data))
    # print('max : ', np.max(data))

    # this is not needed !
    data -= data.min()
    data /= data.max()

    # print('min : ', np.min(data))
    # print('max : ', np.max(data))

    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    
    
    
    if grayscale == True:
        plt.imshow(data, cmap=cm.Greys_r)
    else:
        plt.imshow(data)

    plt.axis('off')

    if filename is None:
        plt.show()
        # plt.draw()
    else:
        plt.savefig(filename, format='png')


def get_subsample(X, y, nc, C=10):
    # nc : number of samples per classes
    G_list = []
    L_list = []
    for c in range(0,C):
        inds_c = np.where(y == c)
        
        inds_c = inds_c[0]
        
        inds_c = np.random.permutation(inds_c)

        G = X[inds_c]
        L = y[inds_c]

        G = G[0:nc]
        L = L[0:nc]

        G_list.append(G)
        L_list.append(L)

    X_sub = G_list[0]
    y_sub = L_list[0]
    for c in range(1,C):
        X_sub = np.concatenate((X_sub, G_list[c]), axis=0)
        y_sub = np.concatenate((y_sub, L_list[c]), axis=0)

    return X_sub, y_sub

# this procedure only works for the rotated MNIST dataset that we provide
def construct_pair(X_list): 
    n_dom = len(X_list)
    X_in = np.vstack(X_list)
    X_outs = []
    for i in range(0, n_dom):
        X = X_list[i]

        Z_list = []
        for j in range(0, n_dom):
            Z_list.append(X)

        Z = np.vstack(Z_list)

        X_outs.append(Z)

        
    
    return X_in, X_outs
