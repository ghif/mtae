# keras 
from keras.models import Sequential
from keras.utils import np_utils, generic_utils
from keras.optimizers import RMSprop, SGD
from keras.layers.core import Dense, Activation, AutoEncoder
from keras.regularizers import activity_l1, l2
from keras.preprocessing.image import ImageDataGenerator

from myutils import *

class Autoencoder(object):
	def __init__(self, n_in, n_hid, 
		lr=1e-2, l2reg=3e-6, corruption_level=0.3, act='sigmoid'):
		self.lr = lr
		self.l2reg = l2reg
		self.corruption_level = corruption_level

		self.ae = Sequential()

		encoder = Sequential()
		encoder.add(Dense(n_in, n_hid, init='uniform', W_regularizer=l2(l2reg)))
		encoder.add(Activation(act))

		decoder = Sequential()
		decoder.add(Dense(n_hid, n_in, init='uniform', W_regularizer=l2(l2reg)))
		decoder.add(Activation(act))

		self.ae.add(AutoEncoder(encoder=encoder, decoder=decoder,
		    output_reconstruction=True))

		opt = RMSprop(lr=lr, rho=0.9, epsilon=1e-6)
		self.ae.compile(loss='mean_squared_error', optimizer=opt)


	def train(self, X_in, X_out, n_epoch=100, batch_size=32, filter_imgfile=None, recon_imgfile=None, verbose=True):

		gdatagen = ImageDataGenerator(
		    featurewise_center=False, # set input mean to 0 over the dataset
		    samplewise_center=False, # set each sample mean to 0
		    featurewise_std_normalization=False, # divide inputs by std of the dataset
		    samplewise_std_normalization=False, # divide each input by its std
		    zca_whitening=False # apply ZCA whitening
		)

		e = 0
		self.rec_losses = []
		while e < n_epoch:
			e += 1
			if verbose:
				print('-'*40)
				print('Epoch', e)
				print('-'*40)

			if verbose:
				progbar = generic_utils.Progbar(X_in.shape[0])

			for X_batch, Y_batch in gdatagen.flow(X_in, X_out, batch_size=batch_size):
			    X_batch = get_corrupted_output(X_batch, corruption_level=self.corruption_level)
			    train_score = self.ae.train_on_batch(X_batch, Y_batch)
			    if verbose:
			    	progbar.add(X_batch.shape[0], values=[("train generative loss", train_score)])

			# Evaluate
			self.loss = self.ae.evaluate(X_in, X_out, batch_size=1024, verbose=0)


			if filter_imgfile is not None:
				# visualize the weights
				W0 = self.ae.get_weights()[0]
				show_images(np.transpose(W0[:,0:100],(1,0)), grayscale=True, filename=filter_imgfile)

			if recon_imgfile is not None:
				# AE recontruction

				# Get random samples
				idx = np.random.permutation(X.shape[0])
				idx = idx[:100]
				Xs = X[idx]

				# Reconstruct input
				Xr = self.ae.predict(Xs)
				show_images(Xr, grayscale=True, filename=recon_imgfile)



