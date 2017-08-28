from keras import optimizers
from keras.layers import Input, Dense, Convolution3D, MaxPooling3D, ZeroPadding3D
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Dropout, Flatten, BatchNormalization, LeakyReLU, Reshape
from keras.layers.merge import concatenate
from keras.layers.wrappers import TimeDistributed
from keras.models import Model, model_from_json
from keras.callbacks import EarlyStopping

import numpy as np


class SpeechEnhancementNet:

	def __init__(self, model):
		self._model = model

	@staticmethod
	def build_video_model(video_input):
		x = ZeroPadding3D(padding=(1, 2, 2), name='v-zero1')(video_input)
		x = Convolution3D(32, (3, 5, 5), strides=(1, 2, 2), kernel_initializer='he_normal', name='v-conv1')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)
		x = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='v-max1')(x)
		x = Dropout(0.25)(x)

		x = ZeroPadding3D(padding=(1, 2, 2), name='v-zero2')(x)
		x = Convolution3D(64, (3, 5, 5), strides=(1, 1, 1), kernel_initializer='he_normal', name='v-conv2')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)
		x = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='v-max2')(x)
		x = Dropout(0.25)(x)

		x = ZeroPadding3D(padding=(1, 1, 1), name='v-zero3')(x)
		x = Convolution3D(128, (3, 3, 3), strides=(1, 1, 1), kernel_initializer='he_normal', name='v-conv3')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)
		x = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='v-max3')(x)
		x = Dropout(0.25)(x)

		x = TimeDistributed(Flatten(), name='v-time')(x)

		x = Dense(1024, kernel_initializer='he_normal', name='v-dense1')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)
		x = Dropout(0.25)(x)

		x = Dense(1024, kernel_initializer='he_normal', name='v-dense2')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)
		x = Dropout(0.25)(x)

		x = Flatten()(x)

		x = Dense(2048, kernel_initializer='he_normal', name='v-dense3')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)
		x = Dropout(0.25)(x)

		x = Dense(2048, kernel_initializer='he_normal', name='v-dense4')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)
		x = Dropout(0.25)(x)

		return x

	@staticmethod
	def build_audio_model(audio_input):
		# x = ZeroPadding2D(padding=(2, 2), name='a-zero1')(audio_input)
		x = Convolution2D(32, (3, 3), kernel_initializer='he_normal', name='a-conv1')(audio_input)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)
		x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='a-max1')(x)
		x = Dropout(0.25)(x)

		# x = ZeroPadding2D(padding=(1, 2, 2), name='a-zero2')(x)
		x = Convolution2D(64, (3, 3), kernel_initializer='he_normal', name='a-conv2')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)
		x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='a-max2')(x)
		x = Dropout(0.25)(x)

		# x = ZeroPadding2D(padding=(1, 1, 1), name='a-zero3')(x)
		x = Convolution2D(128, (3, 3), kernel_initializer='he_normal', name='a-conv3')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)
		x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='a-max3')(x)
		x = Dropout(0.25)(x)

		# x = ZeroPadding2D(padding=(1, 1, 1), name='a-zero4')(x)
		x = Convolution2D(128, (3, 3), kernel_initializer='he_normal', name='a-conv4')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)
		x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='a-max4')(x)
		x = Dropout(0.25)(x)

		x = Flatten()(x)

		x = Dense(2048, kernel_initializer='he_normal', name='a-dense1')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)
		x = Dropout(0.25)(x)

		x = Dense(2048, kernel_initializer='he_normal', name='a-dense2')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)
		x = Dropout(0.25)(x)

		return x

	@classmethod
	def build(cls, video_shape, audio_spectrogram_shape):
		video_input = Input(shape=video_shape)

		# append channels axis
		extended_audio_spectrogram_shape = list(audio_spectrogram_shape)
		extended_audio_spectrogram_shape.append(1)

		audio_input = Input(shape=extended_audio_spectrogram_shape)

		x_video = cls.build_video_model(video_input)
		x_audio = cls.build_audio_model(audio_input)

		x = concatenate([x_video, x_audio])

		x = Dense(4096, kernel_initializer='he_normal', name='av-dense1')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)
		x = Dropout(0.25)(x)

		x = Dense(4096, kernel_initializer='he_normal', name='av-dense2')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)
		x = Dropout(0.25)(x)

		x = Dense(
			audio_spectrogram_shape[0] * audio_spectrogram_shape[1],
			activation='sigmoid', kernel_initializer='he_normal', name='av-dense-output'
		)(x)

		output = Reshape(audio_spectrogram_shape, name='av-reshape-output')(x)

		model = Model(inputs=[video_input, audio_input], outputs=output)
		model.summary()

		return SpeechEnhancementNet(model)

	@staticmethod
	def load(model_cache_path, weights_cache_path):
		with open(model_cache_path, "r") as model_fd:
			model = model_from_json(model_fd.read())

		model.load_weights(weights_cache_path)

		return SpeechEnhancementNet(model)

	def train(self, x_video, x_audio, y):
		x_audio = np.expand_dims(x_audio, -1) # append channels axis

		optimizer = optimizers.adam(lr=0.001, decay=1e-6)
		self._model.compile(loss='binary_crossentropy', optimizer=optimizer)

		early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=20, verbose=1)

		self._model.fit(
			[x_video, x_audio], y,
			batch_size=32, epochs=200, validation_split=0.2, verbose=1,
			callbacks=[early_stopping_callback]
		)

	def predict(self, x_video, x_audio):
		x_audio = np.expand_dims(x_audio, -1) # append channels axis

		y = self._model.predict([x_video, x_audio])
		return y

	def dump(self, model_cache_path, weights_cache_path):
		with open(model_cache_path, 'w') as model_fd:
			model_fd.write(self._model.to_json())

		self._model.save_weights(weights_cache_path)

	def _get_layer_names(self):
		return [layer.name for layer in self._model.layers]
