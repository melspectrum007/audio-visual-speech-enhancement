import numpy as np
import keras.backend as tf

from keras.layers import *
from keras.models import *



class WavenetVocoder(object):

	def __init__(self, num_upsample_channels, num_dilated_blocks):
		self.num_upsample_channels = num_upsample_channels
		self.num_dilated_blocks = num_dilated_blocks

	def build_dilated_conv_block(self, input_shape, kernel_size, dilation, num_dilated_filters, num_skip_filters, number=None):
		layer_input = Input(input_shape)

		filters = Conv1D(num_dilated_filters, kernel_size, padding='same', dilation_rate=dilation, activation='tanh')(layer_input)
		gate = Conv1D(num_dilated_filters, kernel_size, padding='same', dilation_rate=dilation, activation='sigmoid')(layer_input)

		skip = Conv1D(num_skip_filters, kernel_size=1, padding='same')(Multiply()([filters, gate]))

		output = Add()([layer_input, skip])

		if number:
			name = 'dilated_block_' + str(number)
			model = Model(layer_input, outputs=[output, skip], name=name)
		else:
			model = Model(layer_input, outputs=[output, skip])

		return model

	def build_upsample_net(self, input_shape):
		layer_input = Input(input_shape)
		extended = Lambda(lambda a: tf.expand_dims(layer_input, axis=1))(layer_input)

		x = Deconv2D(self.num_upsample_channels, kernel_size=(1, 4), strides=(1, 4), padding='same')(extended)
		x = Deconv2D(self.num_upsample_channels, kernel_size=(1, 4), strides=(1, 4), padding='same')(x)
		x = Deconv2D(self.num_upsample_channels, kernel_size=(1, 10), strides=(1, 10), padding='same')(x)
		# x = Lambda(lambda a: tf.squeeze(x, axis=1))(x)
		x = Reshape((3200,80))(x)

		model = Model(layer_input, x, name='Upsample_Net')
		model.summary()
		return model

	def build(self, spec_shape):

		spectrogram = Input(spec_shape)

		upsample_net = self.build_upsample_net(spec_shape)

		out = upsample_net(spectrogram)
		out = Conv1D(16, 1)(out)

		skips = []
		for i in range(self.num_dilated_blocks):
			out, skip = self.build_dilated_conv_block((3200, 16), 5, 2**(i % 10), 16, 16, number=i+1)(out)
			skips.append(skip)

		stack = Lambda(lambda x: tf.stack(x, -1))(skips)
		stack = Reshape((3200, 16 * self.num_dilated_blocks))(stack)
		stack = Activation('relu')(stack)

		MoL_params = Conv1D(30, 1)(stack)

		model = Model(spectrogram, outputs=[MoL_params], name='Vocoder')
		model.summary()

		return model


def loss(y_true, y_pred):
	means = y_pred[:,:10]
	sigmas = y_pred[:,10:20]
	weights = y_pred[:,20:]

	args = (y_true - means) / (2 * sigmas)
	args = tf.stack([args, -args], axis=-1)
	loglogistic = -tf.log(sigmas) -2 * tf.logsumexp(args, axis=-1)
	logs = loglogistic + tf.log(weights)
	loglikelihood = tf.sum(tf.logsumexp(logs, axis=-1), axis=-1)

	return loglikelihood




if __name__ == '__main__':
	net = WavenetVocoder(80, 3)
	# net.build_upsample_net((20, 80)).summary()
	net.build((20, 80))


