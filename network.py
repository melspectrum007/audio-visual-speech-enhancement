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
	def __build_res_block(cls, input_shape, vid_shape, num_filters, kernel_size, number=None):
		input_spec = Input(input_shape)
		vid_input = Input(vid_shape)

		x = Concatenate()([input_spec, vid_input])

		x = Conv1D(num_filters, kernel_size, padding='same')(x)
		x = BatchNormalization()(x)
		x = Activation('relu')(x)
		x = Conv1D(num_filters, kernel_size, padding='same')(x)

		x = Add()([input_spec, x])

		if number is not None:
			model = Model([input_spec, vid_input], x, name='res_block_' + str(number))
			if number == 1:
				print 'Res Block'
				model.summary()
		else:
			model = Model([input_spec, vid_input], x)

		return model

	@staticmethod
	def __build_video_encoder(video_shape):
		video_input = Input(shape=video_shape)

		x = Lambda(lambda a: K.expand_dims(a, -1))(video_input)

		x = TimeDistributed(Conv2D(80, (5, 5), padding='same'))(x)
		x = TimeDistributed(BatchNormalization())(x)
		x = TimeDistributed(LeakyReLU())(x)
		x = TimeDistributed(MaxPool2D(strides=(2, 2), padding='same'))(x)

		x = TimeDistributed(Conv2D(80, (5, 5), padding='same'))(x)
		x = TimeDistributed(BatchNormalization())(x)
		x = TimeDistributed(LeakyReLU())(x)
		x = TimeDistributed(MaxPool2D(strides=(2, 2), padding='same'))(x)

		x = TimeDistributed(Conv2D(80, (3, 3), padding='same'))(x)
		x = TimeDistributed(BatchNormalization())(x)
		x = TimeDistributed(LeakyReLU())(x)
		x = TimeDistributed(MaxPool2D(strides=(2, 2), padding='same'))(x)

		x = TimeDistributed(Conv2D(80, (3, 3), padding='same'))(x)
		x = TimeDistributed(BatchNormalization())(x)
		x = TimeDistributed(LeakyReLU())(x)
		x = TimeDistributed(MaxPool2D(strides=(2, 2), padding='same'))(x)

		x = TimeDistributed(Conv2D(80, (3, 3), padding='same'))(x)
		x = TimeDistributed(BatchNormalization())(x)
		x = TimeDistributed(LeakyReLU())(x)
		x = TimeDistributed(MaxPool2D(strides=(2, 2), padding='same'))(x)

		x = TimeDistributed(Conv2D(80, (3, 3), padding='same'))(x)
		x = TimeDistributed(BatchNormalization())(x)
		x = TimeDistributed(LeakyReLU())(x)
		x = TimeDistributed(MaxPool2D(strides=(2, 2), padding='same'))(x)

		x = TimeDistributed(Flatten())(x)

		x = Conv1D(80, 5, padding='same')(x)
		x = Conv1D(80, 5, padding='same')(x)
		x = Conv1D(80, 5, padding='same')(x)
		x = Conv1D(80, 5, padding='same')(x)

		x = UpSampling1D(4)(x)

		model = Model(inputs=video_input, outputs=x, name='Video_Encoder')
		print 'Video Encoder'
		model.summary()

		return model

	@staticmethod
	def __build_upsample_net(spec_shape):
		spec = Input(spec_shape)

		x = Lambda(lambda a: K.expand_dims(a, axis=1))(spec)

		x = Deconv2D(spec_shape[1], kernel_size=(1, 11), strides=(1, 2), padding='same')(x)
		x = Deconv2D(spec_shape[1], kernel_size=(1, 11), strides=(1, 4), padding='same')(x)
		x = Deconv2D(spec_shape[1], kernel_size=(1, 11), strides=(1, 4), padding='same')(x)
		x = Deconv2D(spec_shape[1], kernel_size=(1, 11), strides=(1, 5), padding='same')(x)

		x = Lambda(lambda a: K.squeeze(a, axis=1))(x)

		model = Model(inputs=spec, outputs=x, name='Upsample_Net')

		print 'Upsample Net'
		model.summary()

		return model

	@staticmethod
	def __build_dilated_conv_block(layer_input_shape, kernel_size, dilation, num_dilated_filters, num_skip_filters,
								   number=None):
		layer_input = Input(layer_input_shape)

		x = Conv1D(num_dilated_filters, kernel_size, padding='same', dilation_rate=dilation)(layer_input)
		x = LeakyReLU()(x)

		# skip = Conv1D(num_skip_filters, kernel_size=1, padding='same')(x)
		residual = Conv1D(num_skip_filters, kernel_size=1, padding='same')(x)

		output = Add()([layer_input, residual])

		if number:
			name = 'dilated_block_' + str(number)
			model = Model(inputs=[layer_input], outputs=[output], name=name)
		else:
			model = Model(inputs=[layer_input], outputs=[output])

		if number == 1:
			print 'Dilated Conv Block'
			model.summary()

		return model

	@classmethod
	def build(cls, vid_shape, spec_shape, num_filters, kernel_size, num_blocks, num_gpus, num_probs, model_cache_dir):

		input_vid = Input(vid_shape)
		input_spec = Input(spec_shape)

		vid_encoding = cls.__build_video_encoder(vid_shape)(input_vid)

		spec = Concatenate()([input_spec, vid_encoding])

		spec = Conv1D(num_filters, kernel_size, padding='same')(spec)
		spec = BatchNormalization()(spec)
		spec = Activation('relu')(spec)

		for i in range(num_blocks):
			spec = cls.__build_res_block(spec_shape,
									  vid_shape=spec_shape,
									  num_filters=num_filters,
									  kernel_size=kernel_size,
									  number=i)([spec, vid_encoding])

		waveform_logits = cls.__build_upsample_net(spec_shape)(spec)

		waveform_logits = TimeDistributed(Dense(num_probs))(waveform_logits)

		waveform_shape = (spec_shape[0] * 160, num_probs) if spec_shape[0] is not None else (None, num_probs)

		for i in range(num_blocks):
			waveform_logits = cls.__build_dilated_conv_block(waveform_shape,
															 kernel_size=2,
															 dilation=(2**i) % 10,
															 num_dilated_filters=num_probs,
															 num_skip_filters=num_probs,
															 number=i+1)(waveform_logits)

		if num_gpus > 1:
			with tf.device('/cpu:0'):
				model = Model(inputs=[input_vid, input_spec], outputs=[spec, waveform_logits], name='Net')
				fit_model = multi_gpu_model(model, gpus=num_gpus)
		else:
			model = Model(inputs=[input_vid, input_spec], outputs=[spec, waveform_logits], name='Net')
			fit_model = model

		optimizer = optimizers.Adam(lr=5e-4)
		fit_model.compile(loss=['mean_squared_error', 'categorical_crossentropy'], optimizer=optimizer)

		print 'Net'
		model.summary()

		return SpeechEnhancementNetwork(model, fit_model, num_gpus, model_cache_dir)

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
	net = SpeechEnhancementNetwork.build((None, 128, 128), (None, 80), num_filters=80, num_blocks=15, kernel_size=5, num_gpus=1,
										 model_cache_dir=None, num_probs=256)