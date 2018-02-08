import numpy as np
import keras.backend as tf

from keras.layers import *
from keras.models import *
from network import ModelCache
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping


class WavenetVocoder(object):

	def __init__(self, num_upsample_channels=None, num_dilated_blocks=None, spec_shape=None):
		if num_upsample_channels is not None:
			self.num_upsample_channels = num_upsample_channels
			self.num_dilated_blocks = num_dilated_blocks
			self.__model = self.build(spec_shape)


	def build_upsample_net(self, input_shape):
		layer_input = Input(input_shape)
		permuted = Permute((2, 1))(layer_input)
		extended = Lambda(lambda a: tf.expand_dims(a, axis=1))(permuted)

		x = Deconv2D(self.num_upsample_channels, kernel_size=(1, 2), strides=(1, 2), padding='same')(extended)
		x = Deconv2D(self.num_upsample_channels, kernel_size=(1, 2), strides=(1, 2), padding='same')(x)
		# x = Deconv2D(self.num_upsample_channels, kernel_size=(1, 10), strides=(1, 10), padding='same')(x)

		x = Lambda(lambda a: tf.squeeze(a, axis=1))(x)

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
		out = Conv1D(16, 1)(out)

		skips = []
		for i in range(self.num_dilated_blocks):
			out, skip = self.build_dilated_conv_block((None, 16), 2, 2**(i % 10), 16, 16, number=i+1)(out)
			skips.append(skip)

		stack = Concatenate()(skips)
		stack = Activation('relu')(stack)

		MoL_params = Conv1D(30, 1)(stack)

		model = Model(spectrogram, outputs=[MoL_params], name='Vocoder')
		optimizer = optimizers.adam(lr=1e-3)
		model.compile(loss=loss, optimizer=optimizer)

		print 'Vocoder'
		model.summary()

		return model

	def train(self, train_enhanced_spectrograms, train_waveforms, val_enhanced_spectrograms, val_waveforms, model_cache_dir):
		train_waveforms = np.c_[train_waveforms, np.zeros((train_waveforms.shape[0], 160))] # todo: fix net size or label size
		val_waveforms = np.c_[val_waveforms, np.zeros((val_waveforms.shape[0], 160))]

		train_waveforms = np.stack([train_waveforms] * 30, -1) * 127.5
		val_waveforms = np.stack([val_waveforms] * 30, -1) * 127.5

		model_cache = ModelCache(model_cache_dir)
		checkpoint = ModelCheckpoint(model_cache.model_path(), verbose=1)

		lr_decay = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0, verbose=1)
		early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=10, verbose=1)

		self.__model.fit(
			x=[train_enhanced_spectrograms], y=[train_waveforms],
			validation_data=([val_enhanced_spectrograms], [val_waveforms]),
			batch_size=1, epochs=100000,
			callbacks=[checkpoint, lr_decay, early_stopping],
			verbose=1
		)

	def predict_one_sample(self, enhanced_spectrograms):
		params = self.__model.predict(enhanced_spectrograms, batch_size=1)
		means = np.squeeze(params[:,:,:10])
		sigmas = np.exp(np.squeeze(np.abs(params[:,:,10:20])))
		weights_logits = np.squeeze(np.abs(params[:,:,20:]))

		weights = np.exp(weights_logits - log_sum_exp(weights_logits))

		column_indices = sample_many_categorical_once(weights)
		rows_indices = np.arange(means.shape[0])

		means = means[rows_indices, column_indices]
		sigmas = sigmas[rows_indices, column_indices]

		return np.random.logistic(means, sigmas) / 127.5

	@staticmethod
	def load(model_cache_dir):
		model_cache = ModelCache(model_cache_dir)
		model = load_model(model_cache.model_path(), custom_objects={'loss':loss, 'tf':tf})
		vocoder = WavenetVocoder()
		vocoder.__model = model

		return vocoder


def loss(y_true, y_pred):
	y_true = tf.expand_dims(y_true[:,:,0])

	means = y_pred[:,:,:10]
	log_sigmas = y_pred[:,:,10:20]
	weights_logits = y_pred[:,:,20:]

	log_weights = weights_logits - tf.logsumexp(weights_logits)

	sigmas = tf.exp(log_sigmas)
	args = (y_true - means) / (2 * sigmas)
	args = tf.stack([args, -args], axis=-1)
	loglogistic = -log_sigmas - 2 * tf.logsumexp(args, axis=-1)
	logs = loglogistic + log_weights
	loglikelihood = tf.sum(tf.logsumexp(logs, axis=-1), axis=-1)

	return -loglikelihood


def sample_many_categorical_once(weights):
	norm_weights = weights / weights.sum(-1)[..., np.newaxis]
	cdf = np.cumsum(norm_weights, -1)
	n = weights.shape[-2] if weights.ndim > 1 else 1
	uniform = np.random.uniform(size=n)
	indices = (cdf <= uniform[:, np.newaxis]).argmin(-1)

	return np.squeeze(indices)


def log_sum_exp(x):
	m = x.max()
	return m + np.log(np.sum(np.exp(x - m)))


if __name__ == '__main__':
	net = WavenetVocoder(16, 15, (80, 20))


