from keras import optimizers
from keras.layers import Input, Dense, Convolution3D, MaxPooling3D, ZeroPadding3D
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Dropout, Flatten, BatchNormalization, LeakyReLU, Reshape
from keras.layers.merge import concatenate
from keras.layers.wrappers import TimeDistributed
from keras.models import Model, model_from_json
from keras.callbacks import Callback, TensorBoard, EarlyStopping, ModelCheckpoint
from keras import backend as K

import numpy as np


class SpeechEnhancementGAN(object):

	def __init__(self, generator, discriminator, adversarial):
		self.__generator = generator
		self.__discriminator = discriminator
		self.__adversarial = adversarial

	@classmethod
	def build(cls, video_shape, audio_spectrogram_shape):
		generator = cls.__build_generator(video_shape, audio_spectrogram_shape)
		discriminator = None# cls.__build_discriminator(audio_spectrogram_shape)
		adversarial = None #cls.__build_adversarial(video_shape, audio_spectrogram_shape, generator, discriminator)

		return SpeechEnhancementGAN(generator, discriminator, adversarial)

	@classmethod
	def __build_generator(cls, video_shape, audio_spectrogram_shape):
		video_input = Input(shape=video_shape)

		# append channels axis
		extended_audio_spectrogram_shape = list(audio_spectrogram_shape)
		extended_audio_spectrogram_shape.append(1)

		audio_input = Input(shape=extended_audio_spectrogram_shape)

		x_video = cls.__build_video_encoder(video_input)
		x_audio = cls.__build_audio_encoder(audio_input)

		x = concatenate([x_video, x_audio])

		x = Dense(4096, kernel_initializer='he_normal', name='g-av-dense1')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)
		x = Dropout(0.25)(x)

		x = Dense(4096, kernel_initializer='he_normal', name='g-av-dense2')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)
		x = Dropout(0.25)(x)

		x = Dense(
			audio_spectrogram_shape[0] * audio_spectrogram_shape[1],
			activation='sigmoid', kernel_initializer='he_normal', name='g-av-dense-output'
		)(x)

		audio_output = Reshape(audio_spectrogram_shape, name='g-av-reshape-output')(x)

		model = Model(inputs=[video_input, audio_input], outputs=audio_output)

		optimizer = optimizers.adam(lr=0.001, decay=1e-6)
		model.compile(loss='binary_crossentropy', optimizer=optimizer)

		model.summary()
		return model

	@classmethod
	def __build_video_encoder(cls, video_input):
		x = ZeroPadding3D(padding=(1, 2, 2), name='g-v-zero1')(video_input)
		x = Convolution3D(32, (3, 5, 5), strides=(1, 2, 2), kernel_initializer='he_normal', name='g-v-conv1')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)
		x = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='g-v-max1')(x)
		x = Dropout(0.25)(x)

		x = ZeroPadding3D(padding=(1, 2, 2), name='g-v-zero2')(x)
		x = Convolution3D(64, (3, 5, 5), strides=(1, 1, 1), kernel_initializer='he_normal', name='g-v-conv2')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)
		x = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='g-v-max2')(x)
		x = Dropout(0.25)(x)

		x = ZeroPadding3D(padding=(1, 1, 1), name='g-v-zero3')(x)
		x = Convolution3D(128, (3, 3, 3), strides=(1, 1, 1), kernel_initializer='he_normal', name='g-v-conv3')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)
		x = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='g-v-max3')(x)
		x = Dropout(0.25)(x)

		x = TimeDistributed(Flatten(), name='g-v-time')(x)

		x = Dense(1024, kernel_initializer='he_normal', name='g-v-dense1')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)
		x = Dropout(0.25)(x)

		x = Dense(1024, kernel_initializer='he_normal', name='g-v-dense2')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)
		x = Dropout(0.25)(x)

		x = Flatten()(x)

		x = Dense(2048, kernel_initializer='he_normal', name='g-v-dense3')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)
		x = Dropout(0.25)(x)

		x = Dense(2048, kernel_initializer='he_normal', name='g-v-dense4')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)
		x = Dropout(0.25)(x)

		return x

	@classmethod
	def __build_audio_encoder(cls, audio_input):
		x = Convolution2D(32, (3, 3), kernel_initializer='he_normal', name='g-a-conv1')(audio_input)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)
		x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='g-a-max1')(x)
		x = Dropout(0.25)(x)

		x = Convolution2D(64, (3, 3), kernel_initializer='he_normal', name='g-a-conv2')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)
		x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='g-a-max2')(x)
		x = Dropout(0.25)(x)

		x = Convolution2D(128, (3, 3), kernel_initializer='he_normal', name='g-a-conv3')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)
		x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='g-a-max3')(x)
		x = Dropout(0.25)(x)

		x = Convolution2D(128, (3, 3), kernel_initializer='he_normal', name='g-a-conv4')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)
		x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='g-a-max4')(x)
		x = Dropout(0.25)(x)

		x = Flatten()(x)

		x = Dense(2048, kernel_initializer='he_normal', name='g-a-dense1')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)
		x = Dropout(0.25)(x)

		x = Dense(2048, kernel_initializer='he_normal', name='g-a-dense2')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)
		x = Dropout(0.25)(x)

		return x

	@classmethod
	def __build_discriminator(cls, audio_spectrogram_shape):
		audio_input = Input(shape=audio_spectrogram_shape)

		x = Convolution2D(32, (3, 3), kernel_initializer='he_normal', name='a-conv1')(audio_input)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)
		x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='a-max1')(x)
		x = Dropout(0.25)(x)

		x = Convolution2D(64, (3, 3), kernel_initializer='he_normal', name='a-conv2')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)
		x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='a-max2')(x)
		x = Dropout(0.25)(x)

		x = Convolution2D(128, (3, 3), kernel_initializer='he_normal', name='a-conv3')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)
		x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='a-max3')(x)
		x = Dropout(0.25)(x)

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

		x = Dense(1024, kernel_initializer='he_normal', name='d-a-dense1')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)
		x = Dropout(0.25)(x)

		x = Dense(1024, kernel_initializer='he_normal', name='d-a-dense2')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)
		x = Dropout(0.25)(x)

		label_output = Dense(1, activation='sigmoid', kernel_initializer='he_normal', name='d-a-dense3')(x)

		model = Model(inputs=audio_input, outputs=label_output)

		optimizer = optimizers.adam(lr=0.001, decay=1e-6)
		model.compile(loss='mean_squared_error', optimizer=optimizer)

		model.summary()
		return model

	@staticmethod
	def __build_adversarial(video_shape, audio_spectrogram_shape, generator, discriminator):
		video_input = Input(shape=video_shape)
		audio_input = Input(shape=audio_spectrogram_shape)

		label_output = discriminator(generator(inputs=[video_input, audio_input]))
		model = Model(inputs=[video_input, audio_input], outputs=label_output)

		optimizer = optimizers.adam(lr=0.001, decay=1e-6)
		model.compile(loss='mean_squared_error', optimizer=optimizer)

		model.summary()
		return model

	def train(self, video, noisy_audio, clean_audio, weights_cache_path, tensorboard_dir, batch_size=32, n_epochs=200, n_epochs_per_model=2):
		noisy_audio = np.expand_dims(noisy_audio, -1)  # append channels axis

		n_samples = video.shape[0]

		tensorboard_callback = TensorBoard(log_dir=tensorboard_dir, histogram_freq=0, write_graph=True, write_images=True)

		# early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=20, verbose=1)

		checkpoint_callback = ModelCheckpoint(weights_cache_path, monitor='loss', mode='min', verbose=1, save_best_only=True, save_weights_only=True)

		self.__generator.fit([video, noisy_audio], clean_audio,
							 batch_size=batch_size, epochs=n_epochs,
							 callbacks=[checkpoint_callback, tensorboard_callback], verbose=1)

		# for e in range(0, n_epochs, n_epochs_per_model):
		# 	print("training (epoch = %d)" % e)

			# self.__generator.fit([video, noisy_audio], clean_audio,
			# 	batch_size=batch_size, epochs=n_epochs_per_model, callbacks=[checkpoint_callback, tensorboard_callback], verbose=1)

			# for layer in self.__discriminator.layers:
			# 	layer.trainable = False

			# self.__adversarial.fit([video, noisy_audio], np.ones(n_samples),
			# 	batch_size=batch_size, epochs=n_epochs_per_model, verbose=1)
			#
			# ind = np.random.permutation(n_samples)
			#
			# discriminator_input = np.concatenate((
			# 	noisy_audio[ind[:(n_samples / 2)]],
			# 	clean_audio[ind[(n_samples / 2):]]
			# ))
			#
			# discriminator_labels = np.concatenate((
			# 	np.zeros(n_samples / 2),
			# 	np.ones(n_samples / 2)
			# ))
			#
			# for layer in self.__discriminator.layers:
			# 	layer.trainable = True
			#
			# self.__discriminator.fit(discriminator_input, discriminator_labels,
			# 	batch_size=batch_size, epochs=n_epochs_per_model, verbose=1)

	# def predict(self, video, noisy_audio):
	# 	noisy_audio = np.expand_dims(noisy_audio, -1)  # append channels axis
	#
	# 	return self.__generator.predict([video, noisy_audio])

	@staticmethod
	def load_generator(model_cache_path, weights_cache_path):
		with open(model_cache_path, "r") as model_fd:
			model = model_from_json(model_fd.read())

		model.load_weights(weights_cache_path)

		return model

	def save_generator_model(self, model_cache_path):
		with open(model_cache_path, 'w') as model_fd:
			model_fd.write(self.__generator.to_json())

	def save_generator_weights(self, weights_cache_path):
		self.__generator.save_weights(weights_cache_path)
