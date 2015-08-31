# Model
from keras.models import Sequential
from keras.utils import np_utils, generic_utils
from keras.optimizers import RMSprop
from keras.layers.core import Dense, Activation, AutoEncoder
from keras.regularizers import activity_l1, l2

# Preprocessing
from keras.preprocessing.image import ImageDataGenerator
from sklearn import preprocessing

# Utils
import numpy as np
import sys
import cPickle as pickle
import gzip
from time import strftime
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib


def load_rotated_mnist(datapath='MNIST256_6rotations.pkl.gz',left_out_idx=0):
	domains = pickle.load(gzip.open(datapath,'rb'))

	src_domains = domains[:] # clone the list
	del src_domains[left_out_idx]

	(X_test, y_test) = domains[left_out_idx]
	y_test = domains[left_out_idx][1]

	return src_domains, (X_test, y_test)

def get_corrupted_output(X, corruption_level=0.3):
    return np.random.binomial(1, 1-corruption_level, X.shape) * X

def show_filter(X, padsize=1, padval=0, grayscale=False, filename=None, conv=True):
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

    print('min : ', np.min(data))
    print('max : ', np.max(data))

    # this is not needed !
    data -= data.min()
    data /= data.max()

    print('min : ', np.min(data))
    print('max : ', np.max(data))

    
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


if __name__ == '__main__':
	# Params
	n_epoch = 50
	batch_size = 10

	n_in = 256
	n_hid = 500


	src_domains, (X_test, y_test) = load_rotated_mnist()

	(X_train, y_train) = src_domains[0]

	# Autoencoder training
	ae = Sequential()

	encoder = Sequential()
	encoder.add(Dense(n_in, n_hid, init='uniform', activity_regularizer=activity_l1(1e-6)))
	encoder.add(Activation('sigmoid'))

	decoder = Sequential()
	decoder.add(Dense(n_hid, n_in, init='uniform'))
	decoder.add(Activation('sigmoid'))

	ae.add(AutoEncoder(encoder=encoder, decoder=decoder,
	    output_reconstruction=True))

	opt = RMSprop(lr=1e-3, rho=0.9, epsilon=1e-6)
	ae.compile(loss='mean_squared_error', optimizer=opt)

	gdatagen = ImageDataGenerator(
	    featurewise_center=False, # set input mean to 0 over the dataset
	    samplewise_center=False, # set each sample mean to 0
	    featurewise_std_normalization=False, # divide inputs by std of the dataset
	    samplewise_std_normalization=False, # divide each input by its std
	    zca_whitening=False # apply ZCA whitening
	)

	e = 0
	while e < n_epoch:
		e += 1
		print('-'*40)
		print('Epoch', e)
		print('-'*40)

		progbar = generic_utils.Progbar(X_train.shape[0])
		for X_batch, Y_batch in gdatagen.flow(X_train, X_train, batch_size=batch_size, shuffle=True):
		    X_batch = get_corrupted_output(X_batch, corruption_level=0.5)
		    train_score = ae.train_on_batch(X_batch, Y_batch)
		    progbar.add(X_batch.shape[0], values=[("train generative loss", train_score)])


		# visualize the weights
		W0 = ae.get_weights()[0]
		print(W0.shape)
		show_filter(np.transpose(W0[:,0:100],(1,0)), grayscale=True, conv=False, filename='ae_w0.png')

		# # ae1 recontruction
		# show_filter(X_all[0:100], grayscale=True, conv=False, filename='X_all'+str(l)+'_100.png')

		# X_rec = ae1.predict(X_all[0:100])
		# show_filter(X_rec, grayscale=True, conv=False,filename='glorot_ae'+str(l)+'_rec.png')







