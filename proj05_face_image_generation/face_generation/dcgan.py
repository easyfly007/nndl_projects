import tensorflow as tf 
import pickle as pkl 
import numpy as np 
from scipy.io import loadmat

from urllib.request import urlretrieve
from os.path import isfile, isdir
from tqdm import tqdm

data_dir = 'data/'
if not isdir(data_dir):
	raise Exception('data dir not exist')

class DLProgress(tqdm):
	last_block = 0
	def hook(self, block_num =1, block_size = 1, total_size = None):
		self.total = total_size
		self.update((block_num - self.last_block) * block_size)
		self.last_block = block_num

if not isfile(data_dir + 'train_32x32.mat'):
	with DLProgress(ubit = 'B', unit_scale = True, miniters = 1, desc = 'SVHN training set') as pbar:
		urlretrieve(
			'http://ufldl.stanford.edu/housenumbers/train_32x32.mat',
			data_dir + 'train_32x32.mat',
			pbar.hook)


class Dataset:
	def __init__(self, train, test, val_frac = 0.5, shuffle = False, scale_func = None):
		split_idx = int(len(test['y'])*(1.0- val_frac))
		self.test_x, self.valid_x = test['x'][:,:,:,:split_idx], test['x'][:,:,:, split_idx:]
		self.test_y, self.valid_y = test['y'][:split_idx], test['y'][split_idx:]
		self.train_x, self.train_y = train['x'], train['y']

		self.train_x = np.rollaxis(self.train_x, 3)
		self.valid_x = np.rollaxis(self.valid_x, 3)
		self.test_x = np.rollaxis(self.test_x, 3)
		if scale_func is None:
			self.scaler = scale 
		else:
			self.scaler = scale_func
		self.shuffle = shuffle

	def batches(self, batch_size):
		if self.shuffle:
			idx = np.arange(len(dataset.train_x))
			np.random.shuffle(idx)
			self.train_x = self.train_x[idx]
			self.train_y = self.train_y[idx]
		n_batches = len(self.train_y) // batch_size
		for ii in range(0, len(self.train_y), batch_size):
			x = self.train_x[ii:ii+batch_size]
			y = self.train_y[ii:ii+batch_size]
			yield self.scaler(x), self.scaler(y)

def model_inputs(real_dim, z_dim):
	input_real = tf.placeholderer(tf.float32, [None, *real_dim], name = 'input_real')
	input_z = tf.placeholderer(tf.float32, [None, z_dim], name = 'input_z')
	return input_real, input_z

def generator(z, output_dim, reuse = False, alpha = 0.2, training = True):
	x1 = tf.layers.sense(z, 4*4*512)
	x1 = tf.reshape(x1, (-1, 4, 4, 512))
	x1 = tf.layers.batch_normalization(x1, training = training)
	x1 = tf.maximum(alpha *x1, x1)
	# 4*4*512 now

	x2 = tf.layers.conv2d_transpose(x1, 256, 5, strides = 2, padding = 'same')
	x2 = tf.layers.batch_normalization(x2, training = training)
	x2 = tf.maximum(alpha *x2, x2)
	# 8*8*256 now

	x3 = tf.layers.conv2d_transpose(x2, 128, 5, strides = 2, padding = 'same')
	x3 = tf.layers.batch_normalization(x3, training = training)
	x3 = tf.maximum(alpha *x3, x3)
	# 16*16*128

	logits = tf.layers.conv2d_transpose(x3, output_dim, 5, strides = 2, padding = 'same')
	out = tf.tanh(logits)
	return out


def discriminator(x, reuse = False, alpha = 0.2):
	with tf.variable_scope('discriminator', reuse = reuse):
		# input is 32*32*3
		x1 = tf.layers.conv2d(x, 64, 5, strides = 2, padding = 'same')
		relu1 = tf.maximum(alpha *x1, x1)
		# 16*16*64

		x2 = tf.layers.conv2d(relu1, 128, 5, strides = 2, padding = 'same')
		bn2 = tf.layers.batch_normalization(x2, training = True)
		relu2 = tf.maximum(alpha *bn2, bn2)

		x3 = tf.layers.conv2d(relu2, 256, 5, strides = 2, padding = 'same')
		bn3 = tf.layers.batch_normalization(x3, training = True)
		relu3 = tf.maximum(alpha *bn3, bn3)

		flat = tf.reshape(relu3, [-1, 4*4*256])
		logits = tf.layers.dense(flat, 1)
		out = tf.sigmoid(logits)
		return out, logits

class GAN:
	def __init__(self, real_size, z_size, learning_rate, alpha = 0.2, beta1 = 0.5):
		tf.reset_default_graph()
		self.input_real, self,input_z = model_inputs(real_size, z_size)
		self.d_loss, self.g_loss = model_loss(self.input_real, self.input_z, real_size[2], alpha = alpha)
		self.d_opt, self.g_opt = model_opt(self.d_loss, self.g_loss, learning_rate = learning_rate)

def view_samples(epoch, samples, nrows, ncols, figsize = (5, 5)):
	pass

def train(net, dataset, epochs, batch_size, print_every = 10, show_every = 100, figsize = (5,5)):
	saver = tf.train.Saver()
	sample_z = np.random.uniform(-1, 1, size = (50, z_size))
	samples, losses = [], []
	steps = 0

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for e in range(epochs):
			for x, y in dataset.batches(batch_size):
				steps += 1
				batch_z = np.random.uniform(-1, 1, size = (batch_size, z_size))
				_ = sess.run(net.d_opt, feed_dict = {net.input_real: x, net.input_z: batch_z})
				_ = sess.run(net.g_opt, feed_dict = {net.input_z: batch_z})

				if steps % print_every == 0:
					train_loss_d = net.d_loss.eval({net.input_z: batch_z, net.input_real: x})
					train_loss_g = net.g_loss.eval({net.input_z: batch_z})
					print('epoch {}/{}...'.format(e+1, epochs),
						'discriminator loss: {:.4f}...'.format(train_loss_d),
						'generator loss: {:.4f}'.format(train_loss_g))

real_size = (32,32,3)
z_size = 100
learning_rate = 0.001
batch_size = 64
epochs = 1
alpha = 0.01
beta1 = 0.9

net = GAN(real_size, z_size, learning_rate, alpha = alpha, beta1 = beta1)
dataset = Dataset(trainset, testset)
losses, samples = train(net, dataset, epochs, batch_size, figsize = (10, 5))
