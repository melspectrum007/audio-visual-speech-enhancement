import os

from keras import optimizers
from keras.layers import Input, Dense, Convolution2D, MaxPooling3D, Deconvolution2D, Convolution3D, LSTM, Bidirectional
from keras.layers import Dropout, Flatten, BatchNormalization, LeakyReLU, Reshape, Activation, Lambda
from keras.layers.merge import concatenate, add, Multiply
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard

import keras.backend as tf
import numpy as np
import librosa as lb

class SpeechEnhancementNetwork(object):

	def __init__(self, model):
		self.__model = model

	@classmethod
	def __build_encoder(cls, extended_audio_spectrogram_shape, video_shape):
		audio_input = Input(shape=extended_audio_spectrogram_shape)
		video_input = Input(shape=video_shape)

		audio_encoder = cls.__build_audio_encoder(extended_audio_spectrogram_shape)
		audio_embedding = audio_encoder(audio_input)

		video_encoder = cls.__build_video_encoder(video_shape)
		video_embedding = video_encoder(video_input)

		shared_embeding = concatenate([audio_embedding, video_embedding], axis=1)

		model = Model(inputs=[audio_input, video_input], outputs=shared_embeding)
		print 'Encoder'
		model.summary()

		return model

	@staticmethod
	def __build_audio_encoder(extended_audio_spectrogram_shape):
		audio_input = Input(shape=extended_audio_spectrogram_shape)
		x = Convolution2D(32, kernel_size=(5, 5), strides=(4, 1), padding='same')(audio_input)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)

		x = Convolution2D(32, kernel_size=(5, 5), strides=(2, 1), padding='same')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)

		x = Convolution2D(64, kernel_size=(3, 3), strides=(2, 1), padding='same')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)

		x = Convolution2D(64, kernel_size=(3, 3), strides=(2, 1), padding='same')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)

		x = Convolution2D(64, kernel_size=(3, 3), strides=(2, 1), padding='same')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)

		T = extended_audio_spectrogram_shape[1]

		x = Lambda(lambda a: tf.permute_dimensions(a, (0, 1, 3, 2)))(x)
		x = Reshape((-1, T))(x)

		model = Model(inputs=[audio_input], outputs=[x])
		print 'Audio Encoder'
		model.summary()

		return model

	@staticmethod
	def __build_video_encoder(video_shape):
		video_input = Input(shape=video_shape)

		x = Convolution3D(32, kernel_size=(5, 5, 1), padding='same')(video_input)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)
		x = MaxPooling3D(pool_size=(2, 2, 1), strides=(2, 2, 1), padding='same')(x)
		x = Dropout(0.25)(x)

		x = Convolution3D(32, kernel_size=(5, 5, 1), padding='same')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)
		x = MaxPooling3D(pool_size=(2, 2, 1), strides=(2, 2, 1), padding='same')(x)
		x = Dropout(0.25)(x)

		x = Convolution3D(64, kernel_size=(3, 3, 1), padding='same')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)
		x = MaxPooling3D(pool_size=(2, 2, 1), strides=(2, 2, 1), padding='same')(x)
		x = Dropout(0.25)(x)

		x = Convolution3D(64, kernel_size=(3, 3, 1), padding='same')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)
		x = MaxPooling3D(pool_size=(2, 2, 1), strides=(2, 2, 1), padding='same')(x)
		x = Dropout(0.25)(x)

		x = Convolution3D(128, kernel_size=(3, 3, 1), padding='same')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)
		x = MaxPooling3D(pool_size=(2, 2, 1), strides=(2, 2, 1), padding='same')(x)
		x = Dropout(0.25)(x)

		x = Convolution3D(128, kernel_size=(3, 3, 1), padding='same')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)
		x = MaxPooling3D(pool_size=(2, 2, 1), strides=(2, 2, 1), padding='same')(x)
		x = Dropout(0.25)(x)

		x = Convolution3D(128, kernel_size=(2, 2, 1), padding='valid')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)
		x = Dropout(0.25)(x)

		x = Lambda(lambda a: tf.permute_dimensions(a, (0, 1, 2, 4, 3)))(x)
		x = Reshape((-1, video_shape[2]))(x)

		x = Lambda(lambda a: tf.concatenate([tf.repeat(a[:,:,0], 4), tf.repeat(a[:,:,1], 4), tf.repeat(a[:,:,2], 4),
											 tf.repeat(a[:,:,3], 4), tf.repeat(a[:,:,4], 4)], axis=1))(x)
		x = Lambda(lambda a: tf.permute_dimensions(a, (0, 2, 1)))(x)

		model = Model(inputs=video_input, outputs=x)
		print 'Video Encoder'
		model.summary()

		return model

	# @staticmethod
	# def __build_audio_decoder(embedding):
	# 	x = Deconvolution2D(64, kernel_size=(3, 3), strides=(2, 1), padding='same')(embedding)
	# 	x = BatchNormalization()(x)
	# 	x = LeakyReLU()(x)
	#
	# 	x = Deconvolution2D(64, kernel_size=(3, 3), strides=(2, 1), padding='same')(x)
	# 	x = BatchNormalization()(x)
	# 	x = LeakyReLU()(x)
	#
	# 	x = Deconvolution2D(64, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
	# 	x = BatchNormalization()(x)
	# 	x = LeakyReLU()(x)
	#
	# 	x = Deconvolution2D(32, kernel_size=(5, 5), strides=(2, 1), padding='same')(x)
	# 	x = BatchNormalization()(x)
	# 	x = LeakyReLU()(x)
	#
	# 	x = Deconvolution2D(32, kernel_size=(5, 5), strides=(4, 2), padding='same')(x)
	# 	x = BatchNormalization()(x)
	# 	x = LeakyReLU()(x)
	#
	# 	x = Deconvolution2D(1, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
	#
	# 	return x
	#
	# @classmethod
	# def __build_decoder(cls, shared_embedding_size, audio_embedding_shape):
	# 	shared_embedding_input = Input(shape=(shared_embedding_size,))
	#
	# 	x = Dense(shared_embedding_size)(shared_embedding_input)
	# 	x = BatchNormalization()(x)
	# 	x = LeakyReLU()(x)
	#
	# 	audio_embedding_size = np.prod(audio_embedding_shape)
	#
	# 	x = Dense(audio_embedding_size)(x)
	# 	x = Reshape(audio_embedding_shape)(x)
	# 	x = BatchNormalization()(x)
	# 	audio_embedding = LeakyReLU()(x)
	#
	# 	audio_output = cls.__build_audio_decoder(audio_embedding)
	#
	# 	model = Model(inputs=shared_embedding_input, outputs=audio_output)
	# 	print 'Decoder'
	# 	model.summary()
	#
	# 	return model

	@staticmethod
	def __build_attention(shared_embeding_shape):
		shared_input = Input(shared_embeding_shape)

		x = Bidirectional(LSTM(256, return_sequences=True))(shared_input)
		x = Bidirectional(LSTM(256, return_sequences=True))(x)

		mask = LSTM(320, activation='sigmoid', return_sequences=True)(x)

		model = Model(inputs=shared_input, outputs=mask)
		print 'Attention'
		model.summary()

		return model


	@classmethod
	def build(cls, audio_spectrogram_shape, video_shape):
		# append channels axis
		extended_audio_spectrogram_shape = list(audio_spectrogram_shape)
		extended_audio_spectrogram_shape.append(1)

		video_shape = list(video_shape)
		video_shape.append(1)

		encoder = cls.__build_encoder(
			extended_audio_spectrogram_shape, video_shape
		)


		attention = cls.__build_attention((20, 448))

		audio_input = Input(shape=extended_audio_spectrogram_shape)
		video_input = Input(shape=video_shape)

		shared_embeding = encoder([audio_input, video_input])

		shared_embeding = Lambda(lambda a: tf.permute_dimensions(a, (0, 2, 1)))(shared_embeding)
		mask = attention(shared_embeding)
		mask = Lambda(lambda a: tf.permute_dimensions(a, (0, 2, 1)))(mask)

		audio_input_squeezed = Lambda(lambda x: tf.squeeze(x, axis=-1))(audio_input)
		audio_input_squeezed = Lambda(lambda x: 10**(x/20))(audio_input_squeezed)
		audio_output = Multiply()([mask, audio_input_squeezed])

		model = Model(inputs=[audio_input, video_input], outputs=audio_output)

		optimizer = optimizers.adam(lr=5e-4)
		model.compile(loss='mean_squared_error', optimizer=optimizer)
		print 'Net'
		model.summary()

		return SpeechEnhancementNetwork(model)

	def train(self, train_mixed_spectrograms, train_video_samples, train_label_spectrograms,
			  validation_mixed_spectrograms, validation_video_samples, validation_label_spectrograms,
			  model_cache_dir, tensorboard_dir=None):

		train_mixed_spectrograms = np.expand_dims(train_mixed_spectrograms, -1)  # append channels axis
		train_video_samples = np.expand_dims(train_video_samples, -1)  # append channels axis
		train_label_spectrograms = lb.db_to_amplitude(train_label_spectrograms)

		validation_mixed_spectrograms = np.expand_dims(validation_mixed_spectrograms, -1)  # append channels axis
		validation_video_samples = np.expand_dims(validation_video_samples, -1)  # append channels axis
		validation_label_spectrograms = lb.db_to_amplitude(validation_label_spectrograms)

		model_cache = ModelCache(model_cache_dir)
		checkpoint = ModelCheckpoint(model_cache.model_path(), verbose=1)

		lr_decay = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0, verbose=1)
		early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=10, verbose=1)

		self.__model.fit(
			x=[train_mixed_spectrograms, train_video_samples],
			y=train_label_spectrograms,

			validation_data=(
				[validation_mixed_spectrograms, validation_video_samples],
				validation_label_spectrograms
			),

			batch_size=16, epochs=1000,
			callbacks=[checkpoint, lr_decay, early_stopping],
			verbose=1
		)

	def predict(self, mixed_spectrograms, video_samples):
		mixed_spectrograms = np.expand_dims(mixed_spectrograms, -1)  # append channels axis
		video_samples = np.expand_dims(video_samples, -1)  # append channels axis
		speech_spectrograms = self.__model.predict([mixed_spectrograms, video_samples])

		return np.squeeze(speech_spectrograms)

	def evaluate(self, mixed_spectrograms, video_samples, speech_spectrograms):
		mixed_spectrograms = np.expand_dims(mixed_spectrograms, -1)  # append channels axis
		speech_spectrograms = np.expand_dims(speech_spectrograms, -1)  # append channels axis
		
		loss = self.__model.evaluate(x=[mixed_spectrograms, video_samples], y=speech_spectrograms)

		return loss

	def save(self, model_cache_dir):
		model_cache = ModelCache(model_cache_dir)

		self.__model.save(model_cache.model_path())

	@staticmethod
	def load(model_cache_dir):
		model_cache = ModelCache(model_cache_dir)
		model = load_model(model_cache.model_path(), custom_objects={'tf':tf})

		return SpeechEnhancementNetwork(model)

	# @staticmethod
	# def batch_stretch_0_to_1(batch):
	# 	orig_shape = batch.shape
	# 	batch = batch.reshape(orig_shape[0], -1)
	# 	stretched = ((batch.T - batch.min(axis=1)) / (batch.max(axis=1) - batch.min(axis=1))).T
	# 	return stretched.reshape(orig_shape)

class ModelCache(object):

	def __init__(self, cache_dir):
		self.__cache_dir = cache_dir

	def model_path(self):
		return os.path.join(self.__cache_dir, "model.h5py")

if __name__ == '__main__':
	net = SpeechEnhancementNetwork.build((320, 20), (128, 128, 5))