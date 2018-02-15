import numpy as np
import keras.backend as K
import tensorflow as tf

from keras.layers import *
from keras.models import *
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.utils import multi_gpu_model

from network import ModelCache

class WavenetVocoder(object):

	def __init__(self, num_upsample_channels=None, num_dilated_blocks=None, num_skip_channels=None, spec_shape=None, kernel_size=2, gpus=1):
		if num_upsample_channels is not None:
			self.num_upsample_channels = num_upsample_channels
			self.num_dilated_blocks = num_dilated_blocks
			self.num_skip_channels = num_skip_channels
			self.kernel_size = kernel_size
			self.gpus = gpus
			self.__model, self.__fit_model = self.build(spec_shape)

	def build_upsample_net(self, input_shape):
		layer_input = Input(input_shape)
		permuted = Permute((2, 1))(layer_input)
		extended = Lambda(lambda a: K.expand_dims(a, axis=1))(permuted)

		x = Deconv2D(self.num_upsample_channels, kernel_size=(1, 2), strides=(1, 2), padding='same')(extended)
		x = Deconv2D(self.num_upsample_channels, kernel_size=(1, 2), strides=(1, 2), padding='same')(x)
		# x = Deconv2D(self.num_upsample_channels, kernel_size=(1, 10), strides=(1, 10), padding='same')(x)

		x = Lambda(lambda a: K.squeeze(a, axis=1))(x)

		x = UpSampling1D(size=40)(x)

		model = Model(layer_input, x, name='Upsample_Net')
		model.summary()
		return model

	@staticmethod
	def build_dilated_conv_block(input_shape, kernel_size, dilation, num_dilated_filters, num_skip_filters, number=None):
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

	def build(self, spec_shape):

		spectrogram = Input(spec_shape)

		upsample_net = self.build_upsample_net(spec_shape)

		out = upsample_net(spectrogram)
		out = Conv1D(self.num_skip_channels, 1)(out)

		skips = []
		for i in range(self.num_dilated_blocks):
			out, skip = self.build_dilated_conv_block((None, self.num_skip_channels), self.kernel_size, 2**(i % 10), self.num_skip_channels,
													  self.num_skip_channels, number=i+1)(out)
			skips.append(skip)

		stack = Add()(skips)
		stack = Activation('relu')(stack)

		stack = Conv1D(256, 1)(stack)
		stack = Activation('relu')(stack)
		stack = Conv1D(256, 1)(stack)
		probs = Activation('softmax')(stack)
		probs = Permute((2,1))(probs)

		optimizer = optimizers.adam(lr=5e-4)

		if self.gpus > 1:
			with tf.device('/cpu:0'):
				model = Model(spectrogram, outputs=[probs], name='Vocoder')
				fit_model = multi_gpu_model(model, gpus=self.gpus)
		else:
			model = Model(spectrogram, outputs=[probs], name='Vocoder')
			fit_model = model

		fit_model.compile(loss='categorical_crossentropy', optimizer=optimizer)

		print 'Vocoder'
		model.summary()

		return model, fit_model

	def train(self, train_enhanced_spectrograms, train_waveforms, val_enhanced_spectrograms, val_waveforms, model_cache_dir):
		train_waveforms = mu_law_quantization(train_waveforms, 256, max_val=32768)
		val_waveforms = mu_law_quantization(val_waveforms, 256, max_val=32768)

		train_labels = one_hot_encoding(train_waveforms, 256)
		val_labels = one_hot_encoding(val_waveforms, 256)

		model_cache = ModelCache(model_cache_dir)

		lr_decay = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0, verbose=1)
		early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=10, verbose=1)

		self.__fit_model.fit(
			x=[train_enhanced_spectrograms], y=[train_labels],
			validation_data=([val_enhanced_spectrograms], [val_labels]),
			batch_size=16, epochs=10000,
			callbacks=[lr_decay, early_stopping],
			verbose=1
		)

	def predict_one_sample(self, enhanced_spectrogams):
		logits = self.__model.predict(enhanced_spectrogams)
		classes = np.argmax(logits, axis=1)
		bins = np.linspace(-1, 1, 256 + 1)[:-1]
		y = bins[classes]
		norm_waveform = np.sign(y) * ((1 + 255) ** np.abs(y) - 1) / 255

		return np.squeeze(norm_waveform) * 32768

	# def predict_one_sample(self, enhanced_spectrograms):
	# 	params = self.__model.predict(enhanced_spectrograms, batch_size=1)
	# 	means = np.squeeze(params[:,:,:10])
	# 	sigmas = np.exp(np.squeeze(np.abs(params[:,:,10:20])))
	# 	weights_logits = np.squeeze(np.abs(params[:,:,20:]))
	#
	# 	weights = np.exp(weights_logits - log_sum_exp(weights_logits))
	#
	# 	column_indices = sample_many_categorical_once(weights)
	# 	rows_indices = np.arange(means.shape[0])
	#
	# 	means = means[rows_indices, column_indices]
	# 	sigmas = sigmas[rows_indices, column_indices]
	#
	# 	return np.random.logistic(means, sigmas) / 127.5

	@staticmethod
	def load(model_cache_dir):
		model_cache = ModelCache(model_cache_dir)
		model = load_model(model_cache.model_path(), custom_objects={'tf':K})
		vocoder = WavenetVocoder()
		vocoder.__model = model

		return vocoder

def mu_law_quantization(x, mu, max_val=None):
	if max_val is None:
		max_val = np.abs(x).max()

	x = np.clip(x, -max_val, max_val - 1)
	x /= max_val

	y = np.sign(x) * np.log(1 + np.abs(x) * (mu - 1)) / np.log(1 + (mu - 1))
	bins = np.linspace(-1, 1, mu + 1)
	return np.digitize(y, bins) - 1


def one_hot_encoding(y, num_classes):
	one_hot = np.zeros((y.shape[0], num_classes, y.shape[1]))
	one_hot[np.arange(one_hot.shape[0])[:, np.newaxis], y, np.arange(one_hot.shape[2])] = 1

	return one_hot


# def loss(y_true, y_pred):
# 	y_true = tf.expand_dims(y_true[:,:,0])
#
# 	means = y_pred[:,:,:10]
# 	log_sigmas = y_pred[:,:,10:20]
# 	weights_logits = y_pred[:,:,20:]
#
# 	log_weights = weights_logits - tf.logsumexp(weights_logits)
#
# 	sigmas = tf.exp(log_sigmas)
# 	args = (y_true - means) / (2 * sigmas)
# 	args = tf.stack([args, -args], axis=-1)
# 	loglogistic = -log_sigmas - 2 * tf.logsumexp(args, axis=-1)
# 	logs = loglogistic + log_weights
# 	loglikelihood = tf.sum(tf.logsumexp(logs, axis=-1), axis=-1)
#
# 	return -loglikelihood


# def sample_many_categorical_once(weights):
# 	norm_weights = weights / weights.sum(-1)[..., np.newaxis]
# 	cdf = np.cumsum(norm_weights, -1)
# 	n = weights.shape[-2] if weights.ndim > 1 else 1
# 	uniform = np.random.uniform(size=n)
# 	indices = (cdf <= uniform[:, np.newaxis]).argmin(-1)
#
# 	return np.squeeze(indices)

def log_sum_exp(x):
	m = x.max()
	return m + np.log(np.sum(np.exp(x - m)))

if __name__ == '__main__':
	net = WavenetVocoder(num_upsample_channels=80, num_dilated_blocks=30, num_skip_channels=256, spec_shape=(80, 20))


