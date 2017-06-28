from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import numpy as np 
from six.moves import xrange

class Seq2SeqModel(object):
	def __init__(self, 
		source_vocab_size,
		target_vocab_size,
		buckets,
		size,
		num_layers,
		max_gradient_norm,
		batch_size,
		learning_rate,
		learning_rate_decay_factor,
		attention = True,
		use_lstm = False,
		num_samples = 512,
		forward_only = False,
		dtype = tf.float32):
		self.source_vocab_size = source_vocab_size
		self.target_vocab_size = target_vocab_size
		self.buckets = buckets
		self.batch_size = batch_size
		self.learning_rate = tf.Variable(
			float(learning_rate), trainable = False, dtype = dtype)
		self.learning_rate_decay_op = self.learning_rate.assign(
			self.learning_rate * learning_rate_decay_factor)
		self.attention = attention
		self.global_step = tf.Variable(0, trainable = False)

		output_projection = None
		softmax_loss_functiton = None
		if num_samples > 0 and num_samples < self.target_vocab_size:
			w_t = tf.get_variable('proj_w', [self.target_vocab_size, size], dtype = dtype)
			w = tf.transpose(w_t)
			b = tf.get_variable('proj_b', [self.target_vocab_size], dtype = dtype)
			output_projection = (w, b)

			def smapled_loss(inputs, labels):
				labels = tf.reshape(labels, [-1, 1])
				local_w_t = tf.cast(w_t, tf.float32)
				local_b = tf.cast(b, tf.float32)
				local_inputs = tf.cast(inputs, tf.float32)
				return tf.cast(tf.nn.sampled_softmax_loss(local_w_t, local_b, local_inputs, labels, 
					num_samples, self.target_vocab_size), dtype)
			softmax_loss_functiton = smapled_loss

		single_cell = tf.nn.rnn_cell.GRUCell(size)
		if use_lstm:
			single_cell = tf.nn.rnn_cell.LSTMCell(size)
		single_cell = tf.nn.rnn_cell.DropoutWrapper(single_cell, output_keep_prob = 0.75)
		cell = single_cell
		if num_layers > 1:
			cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] *num_layers)

		# the seq2seq function: we use embedding for the input and attention
		def seq2seq_f(encoder_inputs, decoder_inputs, do_decode, attention):
			if attention:
				return tf.nn.seq2seq.embedding_attention_seq2seq(
					encoder_inputs,
					decoder_inputs,
					cell,
					num_encoder_symbols = source_vocab_size,
					num_decoder_symbols = target_vocab_size,
					embedding_size = size,
					output_projection = output_projection,
					feed_previous = do_decode,
					dtype = dtype)
			return tf.nn.seq2seq.embedding_rnn_seq2seq(
				encoder_inputs,
				decoder_inputs,
				cell,
				num_encoder_symbols = source_vocab_size,
				num_decoder_symbols = target_vocab_size,
				embedding_size = size,
				output_projection = output_projection,
				feed_previous = do_decode,
				dtype = dtype)

		# feeds for inputs
		self.encoder_inputs = []
		self.decoder_inputs = []
		self.target_weights = []
		for i in xrange(buckets[-1][0]):
			self.encoder_inputs.append(tf.placeholder(tf.int32, shape = [None], name = 'decoder{0}'.format(i)))
		for i in xrange(buckets[-1][1] + 1):
			self.decoder_inputs.append(tf.placeholder(tf.int32, shape = [None], name = 'decoder{0}'.format(i)))
			self.target_weights.append(tf.placeholder(dtype, shape = [None], name = 'weight{0}'.format(i)))

		# our targets are decoder inputs shifted by one
		targets = [self.decoder_inputs[i+1] for i in xrange(len(self.decoder_inputs) -1)]
		if forward_only:
			self.outputs, self.losses = tf.nn.seq2seq.model_with_buckets(
				self.encoder_inputs, self.decoder_inputs, targets,
				self.target_weights, buckets, lambda x, y:seq2seq_f(x, y, True, self.attention),
				softmax_loss_functiton = softmax_loss_functiton)
			if output_projection is not None:
				for b in xrange(len(buckets)):
					self.outputs[b] = [tf.matmul(output, output_projection[0]) + output_projection[1]
					for output in self.outputs[b]]
		else:
			self.outputs, self.losses = tf.nn.seq2seq.model_with_buckets(
				self.encoder_inputs, self.decoder_inputs. targets,
				self.target_weights, buckets,
				lambda x, y: seq2seq_f(x, y, False, self.attention),
				softmax_loss_functiton = softmax_loss_functiton)

		params = tf.trainable_variables()
		if not forward_only:
			self.gradient_norms = []
			self.updates = []
			opt = tf.train.GradientDescentOptimizer(self.learning_rate)
			for b in xrange(len(buckets)):
				gradients = tf.gradients(self.losses[b], params)
				clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
				self.gradient_norms.append(norm)
				self.updates.append(opt.apply_gradients(zip(clipped_gradients, params), 
					global_step = self.global_step))
		self.saver = tf.trin.Saver(tf.all_variables())

	def step(self, session, encoder_inputs, decoder_inputs, target_weights, bucket_id, forward_only):
		encoder_size, decoder_size = self.buckets[bucket_id]
		if len(encoder_inputs) != encoder_size:
			raise ValueError('encoder length must be equal to the one in bucket,'
				'%d != %d.' %(len(encoder_inputs), encoder_size))
		if len(decoder_inputs) != decoder_size:
			raise ValueError('decoder length must be equal to the one in bucket,'
				'%d != %d.' %(len(decoder_inputs), decoder_size))
			