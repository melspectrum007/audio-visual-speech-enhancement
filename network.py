
from keras import optimizers
from keras.layers import *
from keras.callbacks import *
from keras.models import Model, load_model
from keras.utils import multi_gpu_model

from utils import ModelCache

import keras.backend as K
import tensorflow as tf
import numpy as np

AUDIO_TO_VIDEO_RATIO = 4
BATCH_SIZE = 4

class SpeechEnhancementNetwork(object):

	def __init__(self, model, fit_model=None, num_gpus=None, model_cache_dir=None):
		self.gpus = num_gpus
		self.model_cache = ModelCache(model_cache_dir)
		self.__model = model
		self.__fit_model = fit_model


	@classmethod
	def __build_res_block(cls, input_shape, vid_shape, num_filters, kernel_size, number=None, last=False):
		input_spec = Input(input_shape)
		previous_delta = Input(input_shape)
		previous_features = Input(input_shape)
		vid_input = Input(vid_shape)


		x = Concatenate()([input_spec, vid_input, previous_delta, previous_features])

		delta = Conv1D(num_filters, kernel_size, padding='same')(x)
		delta = BatchNormalization()(delta)
		delta = LeakyReLU()(delta)
		delta = Dropout(0.5)(delta)
		delta = Conv1D(num_filters, kernel_size, padding='same')(delta)

		if not last:
			features = Conv1D(num_filters, kernel_size, padding='same')(x)
			features = BatchNormalization()(features)
			features = LeakyReLU()(features)
			features = Dropout(0.5)(features)

		delta = Add()([previous_delta, delta])

		outputs = [delta, features] if not last else [delta]

		if number is not None:
			model = Model(inputs=[input_spec, vid_input, previous_delta, previous_features], outputs=outputs, name='res_block_' + str(number))
			if number == 1:
				print 'Res Block'
				model.summary()
		else:
			model = Model(inputs=[input_spec, vid_input, previous_delta, previous_features], outputs=outputs)

		return model

	@classmethod
	def build(cls, vid_shape, spec_shape, num_filters, kernel_size, num_blocks, num_gpus, model_cache_dir):

		input_vid = Input(vid_shape)
		input_spec = Input(spec_shape)

		vid_encoding = cls.__build_video_encoder(vid_shape)(input_vid)

		spec = Conv1D(num_filters, kernel_size, padding='same')(input_spec)
		spec = BatchNormalization()(spec)
		spec = LeakyReLU(spec )

		x = Concatenate()([spec, vid_encoding])

		delta = Conv1D(num_filters, kernel_size, padding='same')(x)
		features = Conv1D(num_filters, kernel_size, padding='same')(x)

		for i in range(num_blocks):
			delta, features = cls.__build_res_block(spec_shape,
									  vid_shape=spec_shape,
									  num_filters=num_filters,
									  kernel_size=kernel_size,
									  number=i)([input_spec, vid_encoding, delta, features])

		delta = cls.__build_res_block(spec_shape, spec_shape, num_filters, kernel_size, last=True)([input_spec, vid_encoding, delta, features])

		out = Add()([input_spec, delta])

		if num_gpus > 1:
			with tf.device('/cpu:0'):
				model = Model(inputs=[input_vid, input_spec], outputs=[out], name='Net')
				fit_model = multi_gpu_model(model, gpus=num_gpus)
		else:
			model = Model(inputs=[input_vid, input_spec], outputs=[out], name='Net')
			fit_model = model

		optimizer = optimizers.Adam(lr=5e-4)
		fit_model.compile(loss='mean_squared_error', optimizer=optimizer)

		print 'Net'
		model.summary()

		return SpeechEnhancementNetwork(model, fit_model, num_gpus, model_cache_dir)

	@staticmethod
	def __build_video_encoder(video_shape):
		video_input = Input(shape=video_shape)

		x = Lambda(lambda a: K.expand_dims(a, -1))(video_input)

		x = TimeDistributed(Conv2D(80, (5, 5), padding='same'))(x)
		x = TimeDistributed(BatchNormalization())(x)
		x = TimeDistributed(LeakyReLU())(x)
		x = TimeDistributed(MaxPool2D(strides=(2, 2), padding='same'))(x)
		x = TimeDistributed(Dropout(0.5))(x)

		x = TimeDistributed(Conv2D(80, (5, 5), padding='same'))(x)
		x = TimeDistributed(BatchNormalization())(x)
		x = TimeDistributed(LeakyReLU())(x)
		x = TimeDistributed(MaxPool2D(strides=(2, 2), padding='same'))(x)
		x = TimeDistributed(Dropout(0.5))(x)

		x = TimeDistributed(Conv2D(80, (3, 3), padding='same'))(x)
		x = TimeDistributed(BatchNormalization())(x)
		x = TimeDistributed(LeakyReLU())(x)
		x = TimeDistributed(MaxPool2D(strides=(2, 2), padding='same'))(x)
		x = TimeDistributed(Dropout(0.5))(x)

		x = TimeDistributed(Conv2D(80, (3, 3), padding='same'))(x)
		x = TimeDistributed(BatchNormalization())(x)
		x = TimeDistributed(LeakyReLU())(x)
		x = TimeDistributed(MaxPool2D(strides=(2, 2), padding='same'))(x)
		x = TimeDistributed(Dropout(0.5))(x)

		x = TimeDistributed(Conv2D(80, (3, 3), padding='same'))(x)
		x = TimeDistributed(BatchNormalization())(x)
		x = TimeDistributed(LeakyReLU())(x)
		x = TimeDistributed(MaxPool2D(strides=(2, 2), padding='same'))(x)
		x = TimeDistributed(Dropout(0.5))(x)

		x = TimeDistributed(Conv2D(80, (3, 3), padding='same'))(x)
		x = TimeDistributed(BatchNormalization())(x)
		x = TimeDistributed(LeakyReLU())(x)
		x = TimeDistributed(MaxPool2D(strides=(2, 2), padding='same'))(x)
		x = TimeDistributed(Dropout(0.5))(x)

		x = TimeDistributed(Flatten())(x)

		x = Conv1D(80, 5, padding='same')(x)
		x = TimeDistributed(BatchNormalization())(x)
		x = TimeDistributed(LeakyReLU())(x)

		x = Conv1D(80, 5, padding='same')(x)
		x = TimeDistributed(BatchNormalization())(x)
		x = TimeDistributed(LeakyReLU())(x)

		x = Conv1D(80, 5, padding='same')(x)
		x = TimeDistributed(BatchNormalization())(x)
		x = TimeDistributed(LeakyReLU())(x)

		x = Conv1D(80, 5, padding='same')(x)

		x = UpSampling1D(4)(x)

		model = Model(inputs=video_input, outputs=x, name='Video_Encoder')
		print 'Video Encoder'
		model.summary()

		return model

	def train(self, train_mixed_spectrograms, train_video_samples, train_label_spectrograms,
			  validation_mixed_spectrograms, validation_video_samples, validation_label_spectrograms):

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
		speech_spectrograms = self.__model.predict([video_samples, mixed_spectrograms], batch_size=1)

		return np.squeeze(speech_spectrograms)

	def evaluate(self, mixed_spectrograms, video_samples, speech_spectrograms):
		
		loss = self.__model.evaluate(x=[video_samples, mixed_spectrograms], y=speech_spectrograms, batch_size=1)

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


if __name__ == '__main__':
	# net = SpeechEnhancementNetwork.build((80, None), (128, 128, None), num_gpus=0)
	net = SpeechEnhancementNetwork.build((None, 128, 128), (None, 80), num_filters=80, num_blocks=15, kernel_size=5, num_gpus=1,
										 model_cache_dir=None)