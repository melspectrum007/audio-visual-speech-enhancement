import os

from datetime import datetime
from keras import optimizers
from keras.layers import *
from keras.callbacks import *

from keras.models import Model, load_model
from keras.utils import multi_gpu_model

import keras.backend as K
import tensorflow as tf
import numpy as np

AUDIO_TO_VIDEO_RATIO = 4
BATCH_SIZE = 2

class SpeechEnhancementNetwork(object):

	def __init__(self, model, fit_model=None, num_gpus=None, model_cache_dir=None):
		self.gpus = num_gpus
		self.model_cache = ModelCache(model_cache_dir)
		self.__model = model
		self.__fit_model = fit_model


	@classmethod
	def __build_res_block(cls, input_shape, num_filters, kernel_size, number=None):
		input_spec = Input(input_shape)

		x = Conv1D(num_filters, kernel_size, padding='same')(input_spec)
		x = BatchNormalization()(x)
		x = Activation('relu')(x)
		x = Conv1D(num_filters, kernel_size, padding='same')(x)

		x = Add()([input_spec, x])

		if number is not None:
			model = Model(input_spec, x, name='res_block_' + str(number))
			if number == 1:
				print 'Res Block'
				model.summary()
		else:
			model = Model(input_spec, x)

		return model

	@classmethod
	def build(cls, vid_shape, spec_shape, num_filters, kernel_size, num_blocks, num_gpus, model_cache_dir):

		input_vid = Input(vid_shape)
		input_spec = Input(spec_shape)

		vid_encoding = cls.__build_video_encoder(vid_shape)(input_vid)

		x = Concatenate()([input_spec, vid_encoding])

		x = Conv1D(num_filters, kernel_size, padding='same')(x)
		x = BatchNormalization()(x)
		x = Activation('relu')(x)

		for i in range(num_blocks):
			x = cls.__build_res_block(spec_shape, num_filters, kernel_size, number=i)(x)


		if num_gpus > 1:
			with tf.device('/cpu:0'):
				model = Model(inputs=[input_vid, input_spec], outputs=[x], name='Net')
				fit_model = multi_gpu_model(model, gpus=num_gpus)
		else:
			model = Model(inputs=[input_vid, input_spec], outputs=[x], name='Net')
			fit_model = model

		optimizer = optimizers.Adam(lr=5e-4)
		fit_model.compile(loss='mean_squared_error', optimizer=optimizer)

		print 'Net'
		model.summary()

		return SpeechEnhancementNetwork(model, fit_model, num_gpus, model_cache_dir)

	# @classmethod
	# def __build_encoder(cls, audio_spectrogram_shape, video_shape):
	# 	audio_input = Input(shape=audio_spectrogram_shape)
	# 	video_input = Input(shape=video_shape)
	#
	# 	audio_encoder = cls.__build_audio_encoder(audio_spectrogram_shape)
	# 	audio_embedding = audio_encoder(audio_input)
	#
	# 	video_encoder = cls.__build_video_encoder(video_shape)
	# 	video_embedding = video_encoder(video_input)
	#
	# 	shared_embeding = Concatenate()([audio_embedding, video_embedding])
	#
	# 	model = Model(inputs=[audio_input, video_input], outputs=shared_embeding, name='Encoder')
	# 	print 'Encoder'
	# 	model.summary()
	#
	# 	return model
	#
	# @staticmethod
	# def __build_audio_encoder(audio_spectrogram_shape):
	# 	audio_input = Input(shape=audio_spectrogram_shape)
	#
	# 	x = Permute((2,1))(audio_input)
	#
	# 	x = Conv1D(64, kernel_size=5, padding='same')(x)
	# 	x = BatchNormalization()(x)
	# 	x = LeakyReLU()(x)
	#
	# 	x = Conv1D(64, kernel_size=5, padding='same')(x)
	# 	x = BatchNormalization()(x)
	# 	x = LeakyReLU()(x)
	#
	# 	x = Conv1D(128, kernel_size=3, padding='same')(x)
	# 	x = BatchNormalization()(x)
	# 	x = LeakyReLU()(x)
	#
	# 	x = Conv1D(128, kernel_size=3, padding='same')(x)
	# 	x = BatchNormalization()(x)
	# 	x = LeakyReLU()(x)
	#
	# 	x = Conv1D(256, kernel_size=3, padding='same')(x)
	# 	x = BatchNormalization()(x)
	# 	x = LeakyReLU()(x)
	#
	# 	x = Conv1D(256, kernel_size=3, padding='same')(x)
	# 	x = BatchNormalization()(x)
	# 	x = LeakyReLU()(x)
	#
	# 	model = Model(inputs=audio_input, outputs=x, name='Audio_Encoder')
	# 	print 'Audio Encoder'
	# 	model.summary()
	#
	# 	return model

	@staticmethod
	def __build_video_encoder(video_shape):
		video_input = Input(shape=video_shape)

		x = Convolution3D(80, kernel_size=(5, 5, 3), padding='same')(video_input)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)
		x = MaxPooling3D(pool_size=(2, 2, 1), strides=(2, 2, 1), padding='same')(x)
		x = Dropout(0.25)(x)

		x = Convolution3D(80, kernel_size=(5, 5, 3), padding='same')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)
		x = MaxPooling3D(pool_size=(2, 2, 1), strides=(2, 2, 1), padding='same')(x)
		x = Dropout(0.25)(x)

		x = Convolution3D(80, kernel_size=(3, 3, 3), padding='same')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)
		x = MaxPooling3D(pool_size=(2, 2, 1), strides=(2, 2, 1), padding='same')(x)
		x = Dropout(0.25)(x)

		x = Convolution3D(80, kernel_size=(3, 3, 3), padding='same')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)
		x = MaxPooling3D(pool_size=(2, 2, 1), strides=(2, 2, 1), padding='same')(x)
		x = Dropout(0.25)(x)

		x = Convolution3D(80, kernel_size=(3, 3, 3), padding='same')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)
		x = MaxPooling3D(pool_size=(2, 2, 1), strides=(2, 2, 1), padding='same')(x)
		x = Dropout(0.25)(x)

		x = Convolution3D(80, kernel_size=(3, 3, 3), padding='same')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)
		x = MaxPooling3D(pool_size=(2, 2, 1), strides=(2, 2, 1), padding='same')(x)
		x = Dropout(0.25)(x)

		x = Convolution3D(80, kernel_size=(2, 2, 1), padding='valid')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)
		x = Dropout(0.25)(x)

		Squeeze = Lambda(lambda a: K.squeeze(a, axis=1))

		x = Squeeze(Squeeze(x))

		x = UpSampling1D(4)(x)

		model = Model(inputs=video_input, outputs=x, name='Video_Encoder')
		print 'Video Encoder'
		model.summary()

		return model

	# @staticmethod
	# def __build_audio_decoder(spectogram_shape):
	# 	audio_spec = Input(spectogram_shape)
	#
	# 	out_spec = TimeDistributed(Dense(256), input_shape=spectogram_shape)(audio_spec)
	# 	out_spec = TimeDistributed(Dense(80), input_shape=spectogram_shape)(out_spec)
	#
	# 	x = Conv1D(80, kernel_size=5, padding='same', activation='tanh')(out_spec)
	# 	x = BatchNormalization()(x)
	#
	# 	x = Conv1D(80, kernel_size=5, padding='same', activation='tanh')(x)
	# 	x = BatchNormalization()(x)
	#
	# 	x = Conv1D(80, kernel_size=5, padding='same', activation='tanh')(x)
	# 	x = BatchNormalization()(x)
	#
	# 	x = Conv1D(80, kernel_size=5, padding='same', activation='tanh')(x)
	# 	x = BatchNormalization()(x)
	#
	# 	x = Conv1D(80, kernel_size=5, padding='same')(x)
	#
	# 	x = Add()([out_spec, x])
	#
	# 	model = Model(inputs=audio_spec, outputs=x, name='Audio_Decoder')
	# 	print 'Audio Decoder'
	# 	model.summary()
	#
	# 	return model
	#
	# @staticmethod
	# def __build_attention(shared_embeding_shape):
	# 	shared_input = Input(shared_embeding_shape)
	#
	# 	x = Bidirectional(LSTM(256, return_sequences=True))(shared_input)
	# 	# x = Bidirectional(LSTM(256, return_sequences=True))(x)
	#
	# 	mask = LSTM(256, activation='relu', return_sequences=True)(x)
	#
	# 	model = Model(inputs=shared_input, outputs=mask, name='Attention')
	# 	print 'Attention'
	# 	model.summary()
	#
	# 	return model


	# @classmethod
	# def build(cls, audio_spectrogram_shape, video_shape, num_gpus=1):
	# 	# append channels axis
	#
	# 	video_shape = list(video_shape)
	# 	video_shape.append(1)
	#
	# 	encoder = cls.__build_encoder(audio_spectrogram_shape, video_shape)
	# 	attention = cls.__build_attention((None, 384))
	# 	decoder = cls.__build_audio_decoder((None, 256))
	#
	# 	input_spec = Input(shape=audio_spectrogram_shape)
	# 	input_frames = Input(shape=video_shape)
	#
	# 	shared_embeding = encoder([input_spec, input_frames])
	#
	# 	coarse_output_spec = attention(shared_embeding)
	# 	fine_output_spec = decoder(coarse_output_spec)
	#
	# 	fine_output_spec = Permute((2,1))(fine_output_spec)
	# 	# fine_output_spec = Permute((2,1))(coarse_output_spec)
	#
	# 	optimizer = optimizers.adam(lr=5e-4, clipnorm=1.0)
	#
	# 	if num_gpus > 1:
	# 		with tf.device('/cpu:0'):
	# 			model = Model(inputs=[input_spec, input_frames], outputs=[fine_output_spec], name='Net')
	# 			fit_model = multi_gpu_model(model, gpus=num_gpus)
	# 	else:
	# 		model = Model(inputs=[input_spec, input_frames], outputs=[fine_output_spec], name='Net')
	# 		fit_model = model
	#
	# 	fit_model.compile(loss='mean_squared_error', optimizer=optimizer)
	#
	# 	print 'Net'
	# 	model.summary()
	#
	# 	return SpeechEnhancementNetwork(model, fit_model, num_gpus)

	def train(self, train_mixed_spectrograms, train_video_samples, train_label_spectrograms,
			  validation_mixed_spectrograms, validation_video_samples, validation_label_spectrograms):

		# train_video_samples = np.expand_dims(train_video_samples, -1)  # append channels axis
		# validation_video_samples = np.expand_dims(validation_video_samples, -1)  # append channels axis

		SaveModel = LambdaCallback(on_epoch_end=lambda epoch, logs: self.save_model())

		lr_decay = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0, verbose=1)
		early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=10, verbose=1)
		tensorboard = TensorBoard(log_dir=self.model_cache.tensorboard_path(),
								  histogram_freq=10,
								  batch_size=BATCH_SIZE * self.gpus,
								  write_graph=False,
								  write_grads=True)
		self.__fit_model.fit(
			x=[train_video_samples, train_mixed_spectrograms],
			y=[train_label_spectrograms],

			validation_data=(
				[validation_video_samples, validation_mixed_spectrograms],
				[validation_label_spectrograms],
			),

			batch_size=BATCH_SIZE * self.gpus, epochs=1000,
			callbacks=[SaveModel, lr_decay, early_stopping, tensorboard],
			verbose=1
		)

	def predict(self, mixed_spectrograms, video_samples):
		video_samples = np.expand_dims(video_samples, -1)  # append channels axis
		speech_spectrograms = self.__model.predict([mixed_spectrograms], batch_size=BATCH_SIZE)

		return np.squeeze(speech_spectrograms)

	def evaluate(self, mixed_spectrograms, video_samples, speech_spectrograms):
		speech_spectrograms = np.expand_dims(speech_spectrograms, -1)  # append channels axis
		
		loss = self.__model.evaluate(x=[mixed_spectrograms, video_samples], y=speech_spectrograms)

		return loss

	def save_model(self):
		try:
			self.__model.save(self.model_cache.model_path())
			self.__model.save(self.model_cache.model_backup_path())
		except Exception as e:
			print(e)

	# def save(self, model_cache_dir):
	# 	model_cache = ModelCache(model_cache_dir)
	#
	# 	self.__model.save(model_cache.model_path())
	# 	self.__model.save(model_cache.model_backup_path())

	@staticmethod
	def load(model_cache_dir):
		model_cache = ModelCache(model_cache_dir)
		model = load_model(model_cache.model_path(), custom_objects={'tf':K})

		return SpeechEnhancementNetwork(model)


class ModelCache(object):

	def __init__(self, cache_dir):
		self.__cache_dir = cache_dir

	def model_path(self):
		return os.path.join(self.__cache_dir, 'model.h5py')

	def model_backup_path(self):
		return os.path.join(self.__cache_dir, 'model_backup.h5py')

	def tensorboard_path(self):
		return os.path.join(self.__cache_dir, 'tensorboard', datetime.now().strftime('%Y-%m-%d_%H:%M:%S'))

if __name__ == '__main__':
	# net = SpeechEnhancementNetwork.build((80, None), (128, 128, None), num_gpus=0)
	net = SpeechEnhancementNetwork.build((128, 128, None, 1), (None, 80), num_filters=80, num_blocks=15, kernel_size=5, num_gpus=1,
										 model_cache_dir=None)