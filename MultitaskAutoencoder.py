
from Autoencoder import *
from myutils import *

class MultitaskAutoencoder(object):
	def __init__(self, n_in, n_hid, n_dom,
		lr=1e-2, l2reg=3e-6, corruption_level=0.3, act='sigmoid'):
		self.lr = lr
		self.l2reg = l2reg
		self.corruption_level = corruption_level

		self.AEs = []
		for d in range(0, n_dom):
			self.AEs.append(
				Autoencoder(n_in, n_hid, lr=lr,l2reg=l2reg,corruption_level=corruption_level,act=act)
			)

		# shared parameters
		(self.W, self.b) = self.AEs[0].ae.layers[0].encoder.get_weights()

	def train(self, X_list, n_epoch=100, batch_size=32, filter_imgfile=None):
		X_in, X_outs = construct_pair(X_list)
		n_dom = len(X_outs)
		
		self.losses = [None]*n_dom
		for d in range(0, n_dom):
			self.losses[d] = []

		for e in range(0, n_epoch):
			for d in range(0, n_dom):
				# show_images(X_in, grayscale=True,filename='X_in.png')
				# show_images(X_outs[d], grayscale=True, filename='X_out.png')

				# copy the shared parameters to the d-th autoencoder
				self.AEs[d].ae.layers[0].encoder.set_weights((self.W, self.b))

				# training the d-th autoencoder
				self.AEs[d].train(X_in, X_outs[d], n_epoch=1, batch_size=batch_size, verbose=False)
				print(' -- Task-',(d+1),' loss : ',self.AEs[d].loss)

				self.losses[d].append(self.AEs[d].loss)

				# store the learnt parameters to the shared variables
				(self.W, self.b) = self.AEs[d].ae.layers[0].encoder.get_weights()

			if filter_imgfile is not None:
				# visualize the weights
				show_images(np.transpose(self.W[:,0:100],(1,0)), grayscale=True, filename=filter_imgfile)

			print('=== [MTAE] Epoch ',(e+1),'/',n_epoch,' ====')

	



        




