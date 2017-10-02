import os

from keras import optimizers
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D
from keras.layers import Dropout, Flatten, BatchNormalization, LeakyReLU, Reshape
from keras.layers.merge import concatenate
from keras.models import Model, load_model
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint

import numpy as np


class SpeechEnhancementGAN(object):

	def __init__(self, generator, discriminator, adversarial):
		self.__generator = generator
		self.__discriminator = discriminator
		self.__adversarial = adversarial

	@classmethod
	def build(cls, video_shape, audio_spectrogram_shape):
		generator = cls.__build_generator(video_shape, audio_spectrogram_shape)
		discriminator = cls.__build_discriminator(audio_spectrogram_shape)
		adversarial = cls.__build_adversarial(video_shape, audio_spectrogram_shape, generator, discriminator)

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

		x = Dense(2048)(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)
		x = Dropout(0.5)(x)

		x = Dense(2048)(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)
		x = Dropout(0.5)(x)

		x = Dense(2048)(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)
		x = Dropout(0.5)(x)

		# TODO: Deconvolutions?

		x = Dense(audio_spectrogram_shape[0] * audio_spectrogram_shape[1])(x)

		audio_output = Reshape(extended_audio_spectrogram_shape)(x)

		model = Model(inputs=[video_input, audio_input], outputs=audio_output)

		optimizer = optimizers.adam(lr=0.001, decay=1e-6)
		model.compile(loss='mean_squared_error', optimizer=optimizer)

		model.summary()

		return model

	@classmethod
	def __build_video_encoder(cls, video_input):
		x = Convolution2D(128, kernel_size=(5, 5), padding='same')(video_input)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)
		x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
		x = Dropout(0.25)(x)

		x = Convolution2D(128, kernel_size=(5, 5), padding='same')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)
		x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
		x = Dropout(0.25)(x)

		x = Convolution2D(256, kernel_size=(3, 3), padding='same')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)
		x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
		x = Dropout(0.25)(x)

		x = Convolution2D(256, kernel_size=(3, 3), padding='same')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)
		x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
		x = Dropout(0.25)(x)

		x = Convolution2D(512, kernel_size=(3, 3), padding='same')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)
		x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
		x = Dropout(0.25)(x)

		x = Convolution2D(512, kernel_size=(3, 3), padding='same')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)
		x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
		x = Dropout(0.25)(x)

		x = Convolution2D(512, kernel_size=(3, 3), padding='same')(x)

		x = Flatten()(x)

		return x

	@classmethod
	def __build_audio_encoder(cls, audio_input):
		x = Convolution2D(1, kernel_size=(5, 5), strides=(2, 2), padding='same')(audio_input)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)

		x = Convolution2D(1, kernel_size=(4, 4), strides=(1, 1), padding='same')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)

		x = Convolution2D(2, kernel_size=(4, 4), strides=(2, 2), padding='same')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)

		x = Convolution2D(4, kernel_size=(2, 2), strides=(2, 1), padding='same')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)

		x = Convolution2D(4, kernel_size=(2, 2), strides=(2, 1), padding='same')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)

		x = Convolution2D(8, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)

		x = Flatten()(x)

		return x

	@classmethod
	def __build_discriminator(cls, audio_spectrogram_shape):
		# append channels axis
		extended_audio_spectrogram_shape = list(audio_spectrogram_shape)
		extended_audio_spectrogram_shape.append(1)

		audio_input = Input(shape=extended_audio_spectrogram_shape)

		x = Convolution2D(1, kernel_size=(5, 5), strides=(2, 2), padding='same')(audio_input)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)

		x = Convolution2D(1, kernel_size=(4, 4), strides=(1, 1), padding='same')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)

		x = Convolution2D(2, kernel_size=(4, 4), strides=(2, 2), padding='same')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)

		x = Convolution2D(4, kernel_size=(2, 2), strides=(2, 1), padding='same')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)

		x = Convolution2D(4, kernel_size=(2, 2), strides=(2, 1), padding='same')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)

		x = Convolution2D(8, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)

		x = Flatten()(x)

		x = Dense(32)(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)
		x = Dropout(0.25)(x)

		x = Dense(32)(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)
		x = Dropout(0.25)(x)

		label_output = Dense(1, activation='sigmoid')(x)

		model = Model(inputs=audio_input, outputs=label_output)

		optimizer = optimizers.adam(lr=0.001, decay=1e-6)
		model.compile(loss='binary_crossentropy', optimizer=optimizer)

		model.summary()
		return model

	@staticmethod
	def __build_adversarial(video_shape, audio_spectrogram_shape, generator, discriminator, crossentropy_weight=1000):
		video_input = Input(shape=video_shape)

		# append channels axis
		extended_audio_spectrogram_shape = list(audio_spectrogram_shape)
		extended_audio_spectrogram_shape.append(1)

		audio_input = Input(shape=extended_audio_spectrogram_shape)

		generator_output = generator(inputs=[video_input, audio_input])
		label_output = discriminator(generator_output)
		model = Model(inputs=[video_input, audio_input], outputs=[generator_output, label_output])

		optimizer = optimizers.adam(lr=0.001, decay=1e-6)
		model.compile(loss=['mean_squared_error', 'binary_crossentropy'], loss_weights=[1, crossentropy_weight], optimizer=optimizer)

		model.summary()
		return model

	def train(self, video_samples, mixed_spectrograms, speech_spectrograms,
			  model_cache_dir, tensorboard_dir, validation_split=0.1, batch_size=32, n_epochs=200, n_epochs_per_model=2):

		mixed_spectrograms = np.expand_dims(mixed_spectrograms, -1)  # append channels axis
		speech_spectrograms = np.expand_dims(speech_spectrograms, -1)  # append channels axis

		# train_data, validation_data = self.__split_train_validation_data(
		# 	[video_samples, mixed_spectrograms, speech_spectrograms], validation_split
		# )
		#
		# [video_samples_train, mixed_spectrograms_train, speech_spectrograms_train] = train_data
		# [video_samples_validation, mixed_spectrograms_validation, speech_spectrograms_validation] = validation_data
		#
		# n_samples_train = video_samples_train.shape[0]
		# n_samples_validation = video_samples_validation.shape[0]

		# tensorboard_callback = TensorBoard(log_dir=tensorboard_dir, histogram_freq=0, write_graph=True, write_images=True)
		early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=20, verbose=1)

		model_cache = ModelCache(model_cache_dir)
		generator_checkpoint = ModelCheckpoint(model_cache.generator_path(), verbose=1)
		adversarial_checkpoint = ModelCheckpoint(model_cache.adversarial_path(), verbose=1)
		discriminator_checkpoint = ModelCheckpoint(model_cache.discriminator_path(), verbose=1)

		self.__generator.fit([video_samples, mixed_spectrograms], speech_spectrograms,
							 validation_split=0.05, batch_size=32, epochs=400,
							 callbacks=[generator_checkpoint, early_stopping], verbose=1)

		# for e in range(0, n_epochs, n_epochs_per_model):
		# 	print("training (epoch = %d) ..." % e)
		#
		# 	permutation = np.random.permutation(n_samples_train)
		# 	video_samples_subset = video_samples_train[permutation[:(n_samples_train / 2)]]
		# 	mixed_spectrograms_subset = mixed_spectrograms_train[permutation[:(n_samples_train / 2)]]
		# 	real_speech_spectrograms = speech_spectrograms_train[permutation[(n_samples_train / 2):]]
		#
		# 	generated_speech_spectrograms, _ = self.__adversarial.predict([video_samples_subset, mixed_spectrograms_subset])
		#
		# 	discriminator_samples = np.concatenate((generated_speech_spectrograms, real_speech_spectrograms))
		# 	discriminator_labels = np.concatenate((np.zeros(n_samples_train / 2), np.ones(n_samples_train / 2)))
		#
		# 	print("training discriminator ...")
		# 	for layer in self.__discriminator.layers:
		# 		layer.trainable = True
		#
		# 	self.__discriminator.fit(discriminator_samples, discriminator_labels,
		# 		batch_size=batch_size, epochs=n_epochs_per_model,
		# 		callbacks=[discriminator_checkpoint], verbose=1
		# 	)
		#
		# 	print("training adversarial ...")
		# 	for layer in self.__discriminator.layers:
		# 		layer.trainable = False
		#
		# 	self.__adversarial.fit(
		# 		x=[video_samples_train, mixed_spectrograms_train],
		# 		y=[speech_spectrograms_train, np.ones(n_samples_train)],
		#
		# 		validation_data=(
		# 			[video_samples_validation, mixed_spectrograms_validation],
		# 			[speech_spectrograms_validation, np.ones(n_samples_validation)]
		# 		),
		#
		# 		batch_size=batch_size, epochs=n_epochs_per_model,
		# 		callbacks=[adversarial_checkpoint], verbose=1
		# 	)

	def predict(self, video_samples, mixed_spectrograms):
		mixed_spectrograms = np.expand_dims(mixed_spectrograms, -1)  # append channels axis

		speech_spectrograms = self.__generator.predict([video_samples, mixed_spectrograms])
		return np.squeeze(speech_spectrograms)

	@staticmethod
	def load(model_cache_dir):
		model_cache = ModelCache(model_cache_dir)

		generator = load_model(model_cache.generator_path())
		discriminator = load_model(model_cache.discriminator_path())
		adversarial = load_model(model_cache.adversarial_path())

		return SpeechEnhancementGAN(generator, discriminator, adversarial)

	def save(self, model_cache_dir):
		model_cache = ModelCache(model_cache_dir)

		self.__generator.save(model_cache.generator_path())
		self.__discriminator.save(model_cache.discriminator_path())
		self.__adversarial.save(model_cache.adversarial_path())

	@staticmethod
	def __split_train_validation_data(arrays, validation_split):
		n_samples = arrays[0].shape[0]
		permutation = np.random.permutation(n_samples)
		validation_size = int(validation_split * n_samples)
		validation_indices = permutation[:validation_size]
		train_indices = permutation[validation_size:]

		train_arrays = [a[train_indices] for a in arrays]
		validation_arrays = [a[validation_indices] for a in arrays]

		return train_arrays, validation_arrays


class ModelCache(object):

	def __init__(self, cache_dir):
		self.__cache_dir = cache_dir

	def generator_path(self):
		return os.path.join(self.__cache_dir, "generator.h5py")

	def discriminator_path(self):
		return os.path.join(self.__cache_dir, "discriminator.h5py")

	def adversarial_path(self):
		return os.path.join(self.__cache_dir, "adversarial.h5py")
