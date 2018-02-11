import os

from keras import optimizers
from keras.layers import *

from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard

import keras.backend as tf
import numpy as np
import librosa as lb

NUM_FRAMES = 44
NUM_MEL_FREQS = 320

class SpeechEnhancementNetwork(object):

	def __init__(self, model):
		self.__model = model

	def summerize(self):
		self.__model.summary()

	@classmethod
	def __build_encoder(cls, extended_audio_spectrogram_shape, video_shape):
		stft_audio_input = Input(shape=extended_audio_spectrogram_shape)
		video_input = Input(shape=video_shape)

		audio_encoder = cls.__build_audio_encoder(extended_audio_spectrogram_shape)
		stft_audio_embedding = audio_encoder(stft_audio_input)

		video_encoder = cls.__build_video_encoder(video_shape)
		video_embedding = video_encoder(video_input)

		stft_shared_embeding = concatenate([stft_audio_embedding, video_embedding], axis=2)

		model = Model(inputs=[stft_audio_input, video_input], outputs=[stft_shared_embeding])
		print 'Encoder'
		model.summary()

		return model

	@staticmethod
	def __build_audio_encoder(extended_audio_spectrogram_shape):
		audio_input = Input(shape=extended_audio_spectrogram_shape)
		x = Convolution2D(64, kernel_size=(5, 5), strides=(2, 1), padding='same')(audio_input)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)

		x = Convolution2D(64, kernel_size=(5, 5), strides=(2, 1), padding='same')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)

		x = Convolution2D(64, kernel_size=(5, 5), strides=(1, 1), padding='same')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)

		x = Convolution2D(64, kernel_size=(5, 5), strides=(1, 1), padding='same')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)

		x = Convolution2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)

		x = Permute((2, 1, 3))(x)
		x = Conv2D(2048, kernel_size=(1, 80))(x)

		x = Lambda(tf.squeeze, arguments={'axis':2})(x)

		model = Model(inputs=[audio_input], outputs=[x])
		print 'Audio Encoder'
		model.summary()

		return model

	@staticmethod
	def __build_video_encoder(video_shape):
		video_input = Input(shape=video_shape)

		x = Convolution3D(32, kernel_size=(5, 5, 3), padding='same')(video_input)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)
		x = MaxPooling3D(pool_size=(2, 2, 1), strides=(2, 2, 1), padding='same')(x)
		x = Dropout(0.25)(x)

		x = Convolution3D(32, kernel_size=(5, 5, 3), padding='same')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)
		x = MaxPooling3D(pool_size=(2, 2, 1), strides=(2, 2, 1), padding='same')(x)
		x = Dropout(0.25)(x)

		x = Convolution3D(64, kernel_size=(3, 3, 3), padding='same')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)
		x = MaxPooling3D(pool_size=(2, 2, 1), strides=(2, 2, 1), padding='same')(x)
		x = Dropout(0.25)(x)

		x = Convolution3D(64, kernel_size=(3, 3, 3), padding='same')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)
		x = MaxPooling3D(pool_size=(2, 2, 1), strides=(2, 2, 1), padding='same')(x)
		x = Dropout(0.25)(x)

		x = Convolution3D(128, kernel_size=(3, 3, 3), padding='same')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)
		x = MaxPooling3D(pool_size=(2, 2, 1), strides=(2, 2, 1), padding='same')(x)
		x = Dropout(0.25)(x)

		x = Convolution3D(128, kernel_size=(3, 3, 3), padding='same')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)
		x = MaxPooling3D(pool_size=(2, 2, 1), strides=(2, 2, 1), padding='same')(x)
		x = Dropout(0.25)(x)

		x = Convolution3D(256, kernel_size=(2, 2, 3), padding='valid')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)
		x = Dropout(0.25)(x)

		x = Lambda(tf.squeeze, arguments={'axis': 1})(x)
		x = Lambda(tf.squeeze, arguments={'axis': 1})(x)

		x = UpSampling1D(4)(x)

		model = Model(inputs=video_input, outputs=x)
		print 'Video Encoder'
		model.summary()

		return model

	@staticmethod
	def __build_decoder(embedding_shape):

		embedding = Input(embedding_shape)
		embedding_expanded = Reshape((80, 8, 44))(embedding)
		embedding_expanded = Lambda(lambda a: tf.permute_dimensions(a, (0, 1, 3, 2)))(embedding_expanded)


		x = Deconvolution2D(8, kernel_size=(3, 3), strides=(1, 1), padding='same')(embedding_expanded)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)

		x = Deconvolution2D(64, kernel_size=(5, 5), strides=(1, 1), padding='same')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)

		x = Deconvolution2D(64, kernel_size=(5, 5), strides=(1, 1), padding='same')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)

		x = Deconvolution2D(64, kernel_size=(5, 5), strides=(2, 1), padding='same')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)

		x = Deconvolution2D(64, kernel_size=(5, 5), strides=(2, 1), padding='same')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)

		x = Deconvolution2D(2, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)

		x = Activation('sigmoid')(x)

		model = Model(inputs=embedding, outputs=x)
		print 'Decoder'
		model.summary()

		return model

	@staticmethod
	def __build_attention(shared_embeding_shape, hidden_units):
		shared_input = Input(shared_embeding_shape)

		x = Bidirectional(LSTM(hidden_units, return_sequences=True))(shared_input)

		mask = LSTM(640, activation='sigmoid', return_sequences=True)(x)
		# mask = LSTM(640, activation='sigmoid', return_sequences=True, unroll=True)(mask)

		model = Model(inputs=shared_input, outputs=mask)
		print 'Attention'
		model.summary()

		return model


	@classmethod
	def build(cls, audio_shape, video_shape):

		video_shape = list(video_shape)
		video_shape.append(1)

		encoder = cls.__build_encoder(audio_shape, video_shape)
		attention = cls.__build_attention((None, 2304), 1024)
		# decoder = cls.__build_decoder((640, 44))

		Db2amp = Lambda(lambda x: ((tf.exp(tf.abs(x)) - 1) * tf.sign(x)))
		Stack = Lambda(lambda x: tf.stack(x, axis=-1))

		audio_input = Input(shape=audio_shape)
		video_input = Input(shape=video_shape)

		shared_embeding = encoder([audio_input, video_input])
		mask = attention(shared_embeding)
		mask = Permute((2, 1))(mask)
		mask_real = Lambda(lambda a: a[:,:NUM_MEL_FREQS,:])(mask)
		mask_imag = Lambda(lambda a: a[:,NUM_MEL_FREQS:,:])(mask)

		linear_audio = Db2amp(audio_input)
		in_real = Lambda(lambda a: a[:,:,:,0])(linear_audio)
		in_imag = Lambda(lambda a: a[:,:,:,1])(linear_audio)

		output_real = Subtract()([Multiply()([in_real, mask_real]), Multiply()([in_imag, mask_imag])])
		output_imag = Add()([Multiply()([in_imag, mask_real]), Multiply()([in_real, mask_imag])])

		output = Stack([output_real, output_imag])

		model = Model(inputs=[audio_input, video_input], outputs=[output])

		optimizer = optimizers.adam(lr=5e-4)
		model.compile(loss='mean_squared_error', optimizer=optimizer)
		print 'Net'
		model.summary()

		return SpeechEnhancementNetwork(model)

	def train(self, train_mixed, train_video_samples, train_label,
			  validation_mixed, validation_video_samples, validation_label,
			  model_cache_dir):
		train_video_samples = np.expand_dims(train_video_samples, -1)
		validation_video_samples = np.expand_dims(validation_video_samples, -1)

		train_label = np.exp(np.abs(train_label) - 1) * np.sign(train_label)
		validation_label = np.exp(np.abs(validation_label) - 1) * np.sign(validation_label)

		model_cache = ModelCache(model_cache_dir)
		checkpoint = ModelCheckpoint(model_cache.model_path(), verbose=1)

		lr_decay = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0, verbose=1)
		early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=10, verbose=1)

		self.__model.fit(
			x=[train_mixed, train_video_samples],
			y=[train_label],

			validation_data=(
				[validation_mixed, validation_video_samples],
				[validation_label]
			),

			batch_size=16, epochs=1000,
			callbacks=[checkpoint, lr_decay, early_stopping],
			verbose=1
		)

	def predict(self, mixed_stft, video_samples):
		video_samples = np.expand_dims(video_samples, -1)  # append channels axis

		real_enhanced = self.__model.predict([mixed_stft, video_samples])


		return real_enhanced

	# def evaluate(self, mixed_stft, video_samples, speech_spectrograms):
	# 	video_samples = np.expand_dims(video_samples, -1)  # append channels axis
	# 	mixed_real = np.expand_dims(mixed_stft[:, :, :, 0], -1)  # append channels axis
	# 	mixed_imag = np.expand_dims(mixed_stft[:, :, :, 1], -1)  # append channels axis
	#
	# 	mixed_stft = np.expand_dims(mixed_stft, -1)  # append channels axis
	# 	speech_spectrograms = np.expand_dims(speech_spectrograms, -1)  # append channels axis
	#
	# 	loss = self.__model.evaluate(x=[mixed_stft, video_samples], y=speech_spectrograms)
	#
	# 	return loss

	def save(self, model_cache_dir):
		model_cache = ModelCache(model_cache_dir)

		self.__model.save(model_cache.model_path())

	@staticmethod
	def load(model_cache_dir):
		model_cache = ModelCache(model_cache_dir)
		model = load_model(model_cache.model_path(), custom_objects={'tf':tf, 'NUM_FRAMES':NUM_FRAMES})

		return SpeechEnhancementNetwork(model)


class ModelCache(object):

	def __init__(self, cache_dir):
		self.__cache_dir = cache_dir

	def model_path(self):
		return os.path.join(self.__cache_dir, "model.h5py")

if __name__ == '__main__':
	net = SpeechEnhancementNetwork.build((320, None, 2), (128, 128, None))