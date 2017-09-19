from keras import optimizers
from keras.layers import Input, Dense, Convolution3D, MaxPooling3D, ZeroPadding3D
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Dropout, Flatten, BatchNormalization, LeakyReLU, Reshape
from keras.layers.merge import concatenate
from keras.layers.wrappers import TimeDistributed
from keras.models import Model, model_from_json
from keras.callbacks import Callback, TensorBoard, EarlyStopping
from keras import backend as K

import numpy as np


class SpeechEnhancementGAN:

	def __init__(self, video_shape, audio_spectrogram_shape):
		self.video_shape = video_shape
		self.audio_spectrogram_shape = audio_spectrogram_shape

		# append channels axis
		self.extended_audio_spectrogram_shape = list(audio_spectrogram_shape)
		self.extended_audio_spectrogram_shape.append(1)

		self.generator_video_input = Input(shape=video_shape)
		self.generator_audio_input = Input(shape=self.extended_audio_spectrogram_shape)

		self.discriminator_input = Input(shape=self.extended_audio_spectrogram_shape)

		self.adversarial_video_input = Input(shape=video_shape)
		self.adversarial_audio_input = Input(shape=self.extended_audio_spectrogram_shape)

		self.build_generator()
		self.build_discriminator()

		self.build_generator_model()
		self.build_discriminator_model()
		self.build_adversarial_model()

	def build_generator(self):
		x_video = self.build_video_model(self.generator_video_input)
		x_audio = self.build_audio_model(self.generator_audio_input)

		x = concatenate([x_video, x_audio])

		x = Dense(2048, kernel_initializer='he_normal', name='av-dense1')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)
		x = Dropout(0.25)(x)

		x = Dense(2048, kernel_initializer='he_normal', name='av-dense2')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)
		x = Dropout(0.25)(x)

		x = Dense(
			self.extended_audio_spectrogram_shape[0] * self.extended_audio_spectrogram_shape[1],
			activation='sigmoid', kernel_initializer='he_normal', name='av-dense-output'
		)(x)

		self.generator_output = Reshape(self.extended_audio_spectrogram_shape, name='av-reshape-output')(x)

	def build_discriminator(self):
		x = Flatten()(self.discriminator_input)

		# x = Convolution2D(32, (3, 3), kernel_initializer='he_normal', name='a-conv1')(self.discriminator_input)
		# x = BatchNormalization()(x)
		# x = LeakyReLU()(x)
		# x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='a-max1')(x)
		# x = Dropout(0.25)(x)
		#
		# # x = ZeroPadding2D(padding=(1, 2, 2), name='a-zero2')(x)
		# x = Convolution2D(64, (3, 3), kernel_initializer='he_normal', name='a-conv2')(x)
		# x = BatchNormalization()(x)
		# x = LeakyReLU()(x)
		# x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='a-max2')(x)
		# x = Dropout(0.25)(x)
		#
		# # x = ZeroPadding2D(padding=(1, 1, 1), name='a-zero3')(x)
		# x = Convolution2D(128, (3, 3), kernel_initializer='he_normal', name='a-conv3')(x)
		# x = BatchNormalization()(x)
		# x = LeakyReLU()(x)
		# x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='a-max3')(x)
		# x = Dropout(0.25)(x)
		#
		# # x = ZeroPadding2D(padding=(1, 1, 1), name='a-zero4')(x)
		# x = Convolution2D(128, (3, 3), kernel_initializer='he_normal', name='a-conv4')(x)
		# x = BatchNormalization()(x)
		# x = LeakyReLU()(x)
		# x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='a-max4')(x)
		# x = Dropout(0.25)(x)
		#
		# x = Flatten()(x)

		x = Dense(1024, kernel_initializer='he_normal', name='a-dense1')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)
		x = Dropout(0.25)(x)

		x = Dense(1024, kernel_initializer='he_normal', name='a-dense2')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)
		x = Dropout(0.25)(x)

		self.discriminator_output = Dense(1, kernel_initializer='he_normal', name='a-dense3', activation='sigmoid')(x)

	def build_generator_model(self):
		self.GM = Model(inputs=[self.generator_video_input, self.generator_audio_input], outputs=self.generator_output)

		optimizer = optimizers.adam(lr=0.001, decay=1e-6)
		self.GM.compile(loss='mean_squared_error', optimizer=optimizer)

		self.GM.summary()

	def build_discriminator_model(self):
		self.DM = Model(inputs=self.discriminator_input, outputs=self.discriminator_output)

		optimizer = optimizers.adam(lr=0.001, decay=1e-6)
		self.DM.compile(loss='mean_squared_error', optimizer=optimizer)

		self.DM.summary()

	def build_adversarial_model(self):
		self.adversarial_output = self.DM(self.GM(inputs=[self.adversarial_video_input, self.adversarial_audio_input]))
		self.AM = Model(inputs=[self.adversarial_video_input, self.adversarial_audio_input], outputs=self.adversarial_output)

		optimizer = optimizers.adam(lr=0.001, decay=1e-6)
		self.AM.compile(loss='mean_squared_error', optimizer=optimizer)

		self.AM.summary()

	def build_video_model(self, video_input):
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

		x = Dense(1024, kernel_initializer='he_normal', name='v-dense3')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)
		x = Dropout(0.25)(x)

		x = Dense(1024, kernel_initializer='he_normal', name='v-dense4')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)
		x = Dropout(0.25)(x)

		return x

	def build_audio_model(self, audio_input):
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

		x = Dense(1024, kernel_initializer='he_normal', name='a-dense1')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)
		x = Dropout(0.25)(x)

		x = Dense(1024, kernel_initializer='he_normal', name='a-dense2')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)
		x = Dropout(0.25)(x)

		return x

	@staticmethod
	def load_GM(model_cache_path, weights_cache_path):
		with open(model_cache_path, "r") as model_fd:
			model = model_from_json(model_fd.read())

		model.load_weights(weights_cache_path)

		return model

	def train(self, video, noisy_audio, clean_audio, tensorboard_dir, batch_size=16, n_epochs=20, n_epochs_per_model=1):
		clean_audio = np.expand_dims(clean_audio, -1) # append channels axis
		noisy_audio = np.expand_dims(noisy_audio, -1) # append channels axis

		n_samples = video.shape[0]

		tensorboard_callback = TensorBoard(log_dir=tensorboard_dir, histogram_freq=0, write_graph=True, write_images=True)
		iterations_callback = IterationTracker()

		# early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=20, verbose=1)

		for e in range(0, n_epochs, n_epochs_per_model):
			print("training (epoch = %d)" % e)

			self.GM.fit([video, noisy_audio], clean_audio,
						batch_size=batch_size, epochs=n_epochs_per_model, #initial_epoch=e,
						callbacks=[tensorboard_callback, iterations_callback], verbose=1)

			self.AM.fit([video, noisy_audio], np.ones(n_samples),
						batch_size=batch_size, epochs=n_epochs_per_model, #initial_epoch=e,
						callbacks=[iterations_callback], verbose=1)

			ind = np.random.permutation(n_samples)

			discriminator_input = np.concatenate((
				noisy_audio[ind[:(n_samples / 2)]],
				clean_audio[ind[(n_samples / 2):]]
			))

			discriminator_labels = np.concatenate((
				np.zeros(n_samples / 2),
				np.ones(n_samples / 2)
			))

			self.DM.fit(discriminator_input, discriminator_labels,
						batch_size=batch_size, epochs=n_epochs_per_model, #initial_epoch=e,
						callbacks=[iterations_callback], verbose=1)

	def predict(self, video, noisy_audio):
		noisy_audio = np.expand_dims(noisy_audio, -1) # append channels axis

		return self.GM.predict([video, noisy_audio])

	def dump(self, model_cache_path, weights_cache_path):
		with open(model_cache_path, 'w') as model_fd:
			model_fd.write(self.GM.to_json())

		self.GM.save_weights(weights_cache_path)

	# def _get_layer_names(self):
	# 	return [layer.name for layer in self._model.layers]


class IterationTracker(Callback):

	def on_epoch_end(self, epoch, logs={}):
		n_iterations = self.model.optimizer.iterations
		print('\n##### n_iterations: %d #####\n' % K.eval(n_iterations))
