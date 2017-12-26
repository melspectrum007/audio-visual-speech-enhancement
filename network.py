import os

from keras import optimizers
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, Deconvolution2D
from keras.layers import Dropout, Flatten, BatchNormalization, LeakyReLU, Reshape, Activation, Lambda
from keras.layers.merge import concatenate, add, Multiply
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard

import numpy as np
import librosa as lb

class SpeechEnhancementNetwork(object):

	def __init__(self, model):
		self.__model = model

	@classmethod
	def __build_encoder(cls, audio_magphase_shape, video_shape):
		audio_input = Input(shape=audio_magphase_shape)
		video_input = Input(shape=video_shape)

		audio_encoder = cls.__build_audio_encoder(audio_magphase_shape)
		audio_embedding_matrix = audio_encoder(audio_input)
		audio_embedding = Flatten()(audio_embedding_matrix)

		video_encoder = cls.__build_video_encoder(video_shape)
		video_embedding_matrix = video_encoder(video_input)
		video_embedding = Flatten()(video_embedding_matrix)

		x = concatenate([audio_embedding, video_embedding])
		shared_embedding_size = int(x._keras_shape[1] / 4)

		x = Dense(shared_embedding_size)(x)
		x = BatchNormalization()(x)
		shared_embedding = LeakyReLU()(x)

		model = Model(inputs=[audio_input, video_input], outputs=shared_embedding)
		model.summary()

		return model, shared_embedding_size, audio_embedding_matrix.shape[1:].as_list()

	@classmethod
	def __build_decoder(cls, shared_embedding_size, audio_embedding_shape):
		shared_embedding_input = Input(shape=(shared_embedding_size,))

		x = Dense(shared_embedding_size)(shared_embedding_input)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)

		audio_embedding_size = np.prod(audio_embedding_shape)

		x = Dense(audio_embedding_size)(x)
		x = Reshape(audio_embedding_shape)(x)
		x = BatchNormalization()(x)
		audio_embedding = LeakyReLU()(x)

		audio_output = cls.__build_audio_decoder(audio_embedding)

		model = Model(inputs=shared_embedding_input, outputs=audio_output)
		print 'Decoder'
		model.summary()

		return model

	@staticmethod
	def __build_audio_encoder(audio_magphase_shape):
		audio_input = Input(shape=audio_magphase_shape)
		x = Convolution2D(32, kernel_size=(5, 5), strides=(4, 2), padding='same')(audio_input)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)

		x = Convolution2D(32, kernel_size=(5, 5), strides=(2, 1), padding='same')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)

		x = Convolution2D(64, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)

		x = Convolution2D(64, kernel_size=(3, 3), strides=(2, 1), padding='same')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)

		x = Convolution2D(64, kernel_size=(3, 3), strides=(2, 1), padding='same')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)

		model = Model(inputs=[audio_input], outputs=[x])
		print 'Audio Encoder'
		model.summary()

		return model

	@staticmethod
	def __build_audio_decoder(embedding):
		x = Deconvolution2D(64, kernel_size=(3, 3), strides=(2, 1), padding='same')(embedding)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)

		x = Deconvolution2D(64, kernel_size=(3, 3), strides=(2, 1), padding='same')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)

		x = Deconvolution2D(64, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)

		x = Deconvolution2D(32, kernel_size=(5, 5), strides=(2, 1), padding='same')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)

		x = Deconvolution2D(32, kernel_size=(5, 5), strides=(4, 2), padding='same')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)

		x = Deconvolution2D(1, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)

		return x

	@staticmethod
	def __build_video_encoder(video_shape):
		video_input = Input(shape=video_shape)

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

		model = Model(inputs=video_input, outputs=x)
		print 'Video Encoder'
		model.summary()

		return model

	@classmethod
	def build(cls, audio_magphase_shape, video_shape):

		encoder, shared_embedding_size, audio_embedding_shape = cls.__build_encoder(
			audio_magphase_shape, video_shape
		)

		decoder = cls.__build_decoder(shared_embedding_size, (5, 5, 64))

		audio_input = Input(shape=audio_magphase_shape)
		video_input = Input(shape=video_shape)

		# audio_output = decoder(encoder([audio_input, video_input]))
		mask = decoder(encoder([audio_input, video_input]))
		mask = Activation('sigmoid')(mask)
		mask = Reshape(audio_magphase_shape[:-1])(mask)

		audio_spec = Lambda(lambda x: x[:, :, :, 0])(audio_input)
		audio_spec = Lambda(lambda x: 10 ** (x / 20))(audio_spec)

		audio_output = Multiply()([mask, audio_spec])

		model = Model(inputs=[audio_input, video_input], outputs=audio_output)

		optimizer = optimizers.adam(lr=5e-4)
		model.compile(loss='mean_squared_error', optimizer=optimizer)

		model.summary()

		return SpeechEnhancementNetwork(model)

	def train(self, train_mixed_magphases, train_video_samples, train_label_magphases,
			  validation_mixed_magphases, validation_video_samples, validation_label_magphases,
			  model_cache_dir, tensorboard_dir=None):

		train_labels = lb.db_to_amplitude(train_label_magphases[:,:,:,0])
		validation_labels = lb.db_to_amplitude(validation_label_magphases[:,:,:,0])

		train_labels *= np.cos(train_label_magphases[:,:,:,1] - train_mixed_magphases[:,:,:,1])
		validation_labels *= np.cos(validation_label_magphases[:,:,:,1] - validation_mixed_magphases[:,:,:,1])

		model_cache = ModelCache(model_cache_dir)
		checkpoint = ModelCheckpoint(model_cache.model_path(), verbose=1)

		lr_decay = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0, verbose=1)
		early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=10, verbose=1)

		# tensorboard = TensorBoard(log_dir=tensorboard_dir, histogram_freq=0, write_graph=True, write_images=True)

		self.__model.fit(
			x=[train_mixed_magphases, train_video_samples],
			y=train_labels,

			validation_data=(
				[validation_mixed_magphases, validation_video_samples],
				validation_labels
			),

			batch_size=16, epochs=1000,
			callbacks=[checkpoint, lr_decay, early_stopping],
			verbose=1
		)

	def predict(self, mixed_magphases, video_samples):
		speech_magphases = self.__model.predict([mixed_magphases, video_samples])

		return np.squeeze(speech_magphases)

	def evaluate(self, mixed_magphases, video_samples, speech_magphases):
		mixed_magphases = np.expand_dims(mixed_magphases, -1)  # append channels axis
		speech_magphases = np.expand_dims(speech_magphases, -1)  # append channels axis
		
		loss = self.__model.evaluate(x=[mixed_magphases, video_samples], y=speech_magphases)

		return loss

	def save(self, model_cache_dir):
		model_cache = ModelCache(model_cache_dir)

		self.__model.save(model_cache.model_path())

	@staticmethod
	def load(model_cache_dir):
		model_cache = ModelCache(model_cache_dir)
		model = load_model(model_cache.model_path())

		return SpeechEnhancementNetwork(model)

	@staticmethod
	def batch_stretch_0_to_1(batch):
		orig_shape = batch.shape
		batch = batch.reshape(orig_shape[0], -1)
		stretched = ((batch.T - batch.min(axis=1)) / (batch.max(axis=1) - batch.min(axis=1))).T
		return stretched.reshape(orig_shape)

class ModelCache(object):

	def __init__(self, cache_dir):
		self.__cache_dir = cache_dir

	def model_path(self):
		return os.path.join(self.__cache_dir, "model.h5py")
