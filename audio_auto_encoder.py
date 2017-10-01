import argparse
import os
import multiprocess
from datetime import datetime

from keras import optimizers
from keras.layers import Input, Convolution2D, Deconvolution2D, Dense, Flatten, Reshape, Dropout
from keras.layers import BatchNormalization, LeakyReLU

from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint

import numpy as np

from dsp.spectrogram import MelConverter
from mediaio.audio_io import AudioSignal


class AudioAutoEncoder(object):

	def __init__(self, encoder, decoder, auto_encoder):
		self.__encoder = encoder
		self.__decoder = decoder
		self.__auto_encoder = auto_encoder

	@classmethod
	def build(cls, audio_spectrogram_shape):
		# append channels axis
		extended_audio_spectrogram_shape = list(audio_spectrogram_shape)
		extended_audio_spectrogram_shape.append(1)

		audio_input = Input(shape=extended_audio_spectrogram_shape)

		encoder = cls.__build_encoder(extended_audio_spectrogram_shape)
		decoder = cls.__build_decoder(encoder.output_shape[1:])

		auto_encoder = Model(inputs=audio_input, outputs=decoder(encoder(audio_input)))

		optimizer = optimizers.adam(lr=0.001, decay=1e-6)
		auto_encoder.compile(loss='mean_squared_error', optimizer=optimizer)

		auto_encoder.summary()
		return AudioAutoEncoder(encoder, decoder, auto_encoder)

	@staticmethod
	def __build_encoder(extended_audio_spectrogram_shape):
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

		embedding = Convolution2D(8, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)

		model = Model(inputs=audio_input, outputs=embedding)
		model.summary()

		return model

	@staticmethod
	def __build_decoder(embedding_shape):
		embedding = Input(shape=embedding_shape)

		x = Deconvolution2D(8, kernel_size=(1, 1), strides=(1, 1), padding='same')(embedding)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)

		x = Deconvolution2D(4, kernel_size=(2, 2), strides=(2, 1), padding='same')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)

		x = Deconvolution2D(2, kernel_size=(2, 2), strides=(2, 1), padding='same')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)

		x = Deconvolution2D(1, kernel_size=(4, 4), strides=(2, 2), padding='same')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)

		x = Deconvolution2D(1, kernel_size=(4, 4), strides=(1, 1), padding='same')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)

		audio_output = Deconvolution2D(1, kernel_size=(5, 5), strides=(2, 2), padding='same')(x)

		model = Model(inputs=embedding, outputs=audio_output)
		model.summary()

		return model

	def train(self, audio, model_cache_dir):
		extended_audio = np.expand_dims(audio, -1)  # append channels axis

		model_cache = ModelCache(model_cache_dir)
		auto_encoder_checkpoint = ModelCheckpoint(model_cache.auto_encoder_path(), monitor='val_loss', mode='min', verbose=1)

		early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.1, patience=10, verbose=1)

		self.__auto_encoder.fit(extended_audio, extended_audio,
			validation_split=0.1, batch_size=32, epochs=200,
			callbacks=[early_stopping, auto_encoder_checkpoint], verbose=1
		)

	def predict(self, audio):
		extended_audio = np.expand_dims(audio, -1)  # append channels axis

		return np.squeeze(self.__auto_encoder.predict(extended_audio))

	@staticmethod
	def load(model_cache_dir):
		model_cache = ModelCache(model_cache_dir)

		encoder = load_model(model_cache.encoder_path())
		decoder = load_model(model_cache.decoder_path())
		auto_encoder = load_model(model_cache.auto_encoder_path())

		return AudioAutoEncoder(encoder, decoder, auto_encoder)

	def save(self, model_cache_dir):
		model_cache = ModelCache(model_cache_dir)

		self.__encoder.save(model_cache.encoder_path())
		self.__decoder.save(model_cache.decoder_path())
		self.__auto_encoder.save(model_cache.auto_encoder_path())


class ModelCache(object):

	def __init__(self, cache_dir):
		self.__cache_dir = cache_dir

	def encoder_path(self):
		return os.path.join(self.__cache_dir, "encoder.h5py")

	def decoder_path(self):
		return os.path.join(self.__cache_dir, "decoder.h5py")

	def auto_encoder_path(self):
		return os.path.join(self.__cache_dir, "auto_encoder.h5py")


def preprocess_audio_file(path, slice_duration_ms=200):
	print("preprocessing: %s ..." % path)

	audio_signal = AudioSignal.from_wav_file(path)

	n_fft = 640
	hop_length = int(n_fft / 4)

	mel_converter = MelConverter(audio_signal.get_sample_rate(), n_fft, hop_length, n_mel_freqs=80, freq_min_hz=0, freq_max_hz=8000)
	mel_spectrogram = mel_converter.signal_to_mel_spectrogram(audio_signal)

	samples_per_slice = int((float(slice_duration_ms) / 1000) * audio_signal.get_sample_rate())
	spectrogram_samples_per_slice = int(samples_per_slice / hop_length)
	n_slices = int(mel_spectrogram.shape[1] / spectrogram_samples_per_slice)

	slices = [
		mel_spectrogram[:, (i * spectrogram_samples_per_slice):((i + 1) * spectrogram_samples_per_slice)]
		for i in range(n_slices)
	]

	return np.stack(slices)


def reconstruct_signal(signal, spectrograms):
	n_fft = 640
	hop_length = int(n_fft / 4)

	mel_converter = MelConverter(signal.get_sample_rate(), n_fft, hop_length, n_mel_freqs=80, freq_min_hz=0, freq_max_hz=8000)
	_, original_phase = mel_converter.signal_to_mel_spectrogram(signal, get_phase=True)

	spectrogram = np.concatenate(list(spectrograms), axis=1)

	spectrogram_length = min(spectrogram.shape[1], original_phase.shape[1])
	spectrogram = spectrogram[:, :spectrogram_length]
	original_phase = original_phase[:, :spectrogram_length]

	return mel_converter.reconstruct_signal_from_mel_spectrogram(spectrogram, original_phase)


def preprocess(args):
	audio_file_paths = [os.path.join(args.audio_dir, f) for f in os.listdir(args.audio_dir)]

	thread_pool = multiprocess.Pool(8)
	preprocessed_samples = thread_pool.map(preprocess_audio_file, audio_file_paths)

	audio_samples = np.concatenate(preprocessed_samples)

	np.savez(args.preprocessed_blob_path, audio_samples=audio_samples)


def load_preprocessed_samples(preprocessed_blob_path, max_samples=None):
	with np.load(preprocessed_blob_path) as data:
		audio_samples = data["audio_samples"]

	permutation = np.random.permutation(audio_samples.shape[0])
	audio_samples = audio_samples[permutation]

	return audio_samples[:max_samples]


def train(args):
	audio_samples = load_preprocessed_samples(args.preprocessed_blob_path)

	network = AudioAutoEncoder.build(audio_samples.shape[1:])
	network.train(audio_samples, args.model_cache_dir)
	network.save(args.model_cache_dir)


def predict(args):
	prediction_output_dir = os.path.join(args.prediction_output_dir, '{:%Y-%m-%d_%H-%M-%S}'.format(datetime.now()))
	os.mkdir(prediction_output_dir)

	network = AudioAutoEncoder.load(args.model_cache_dir)

	audio_file_paths = [os.path.join(args.audio_dir, f) for f in os.listdir(args.audio_dir)]

	for audio_file_path in audio_file_paths[:3]:
		try:
			print("predicting %s..." % audio_file_path)

			audio_samples = preprocess_audio_file(audio_file_path)
			audio_signal = AudioSignal.from_wav_file(audio_file_path)

			predicted_spectrograms = network.predict(audio_samples)
			predicted_signal = reconstruct_signal(audio_signal, predicted_spectrograms)

			predicted_signal.save_to_wav_file(os.path.join(prediction_output_dir, os.path.basename(audio_file_path)))

		except Exception as e:
			print("failed to predict %s (%s). skipping" % (audio_file_path, e))


def main():
	parser = argparse.ArgumentParser(add_help=False)
	action_parsers = parser.add_subparsers()

	preprocess_parser = action_parsers.add_parser("preprocess")
	preprocess_parser.add_argument("--audio_dir", type=str, required=True)
	preprocess_parser.add_argument("--preprocessed_blob_path", type=str, required=True)
	preprocess_parser.set_defaults(func=preprocess)

	train_parser = action_parsers.add_parser("train")
	train_parser.add_argument("--preprocessed_blob_path", type=str, required=True)
	train_parser.add_argument("--model_cache_dir", type=str, required=True)
	train_parser.set_defaults(func=train)

	predict_parser = action_parsers.add_parser("predict")
	predict_parser.add_argument("--audio_dir", type=str, required=True)
	predict_parser.add_argument("--model_cache_dir", type=str, required=True)
	predict_parser.add_argument("--prediction_output_dir", type=str, required=True)
	predict_parser.set_defaults(func=predict)

	args = parser.parse_args()
	args.func(args)


if __name__ == "__main__":
	main()
