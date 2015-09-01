from Autoencoder import *
from MultitaskAutoencoder import *
import numpy as np

if __name__ == '__main__':
	# Params
	n_epoch = 100
	batch_size = 16

	n_in = 256
	n_hid = 500

	# Load Dataset
	src_domains, (X_test, y_test) = load_rotated_mnist()
	X_list = []
	y_list = []
	for d in range(0, len(src_domains)):
		X, y = src_domains[d]
		X_list.append(X)
		y_list.append(y)

	# print('Training Denoising Autoencoder (DAE) ....')
	# X_train = np.vstack(X_list)

	# ae = Autoencoder(n_in, n_hid, corruption_level=0.3)
	# ae.train(X_train, n_epoch=n_epoch, batch_size=batch_size, 
	# 	filter_imgfile='W0_ae.png',
	# 	recon_imgfile='Xr_ae.png'
	# )


	print('Training Denoising Multitask Autoencoder (D-MTAE)...')
	n_dom = len(src_domains)
	mtae = MultitaskAutoencoder(n_in, n_hid, n_dom, lr=3e-3, l2reg=3e-6, corruption_level=0.3)
	mtae.train(X_list, n_epoch=n_epoch, batch_size=batch_size,
		filter_imgfile='W0_mtae.png'
	)


	






