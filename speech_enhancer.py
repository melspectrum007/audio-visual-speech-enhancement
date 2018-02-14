import argparse
import os
from datetime import datetime
import logging
import pickle

import numpy as np

import data_processor
from dataset import AudioVisualDataset, AudioDataset
from network import SpeechEnhancementNetwork
from shutil import copy2
from mediaio import ffmpeg


def preprocess(args):
	speaker_ids = list_speakers(args)

	speech_subset, noise_file_paths = list_data(
		args.dataset_dir, speaker_ids, args.noise_dirs, max_files=1500, shuffle=True
	)

	samples = data_processor.preprocess_data(speech_subset, noise_file_paths)

	with open(args.preprocessed_blob_path, 'wb') as preprocessed_fd:
		pickle.dump(samples, preprocessed_fd)


def load_preprocessed_samples(preprocessed_blob_paths):
	all_samples = []

	for preprocessed_blob_path in preprocessed_blob_paths:
		print("loading preprocessed samples from %s" % preprocessed_blob_path)

		with open(preprocessed_blob_path, 'rb') as preprocessed_fd:
			samples = pickle.load(preprocessed_fd)

		all_samples += samples

	return all_samples


def merge_training_set(samples, max_sample_slices=None):
	video_samples = np.concatenate([sample.video_samples for sample in samples], axis=0)
	mixed_spectrograms = np.concatenate([sample.mixed_spectrograms for sample in samples], axis=0)
	speech_spectrograms = np.concatenate([sample.speech_spectrograms for sample in samples], axis=0)

	permutation = np.random.permutation(video_samples.shape[0])
	video_samples = video_samples[permutation]
	mixed_spectrograms = mixed_spectrograms[permutation]
	speech_spectrograms = speech_spectrograms[permutation]

	return (
		video_samples[:max_sample_slices],
		mixed_spectrograms[:max_sample_slices],
		speech_spectrograms[:max_sample_slices]
	)


def train(args):
	train_samples = load_preprocessed_samples(args.train_preprocessed_blob_paths)
	train_video_samples, train_mixed_spectrograms, train_speech_spectrograms = merge_training_set(train_samples)

	validation_samples = load_preprocessed_samples(args.validation_preprocessed_blob_paths)
	validation_video_samples, validation_mixed_spectrograms, validation_speech_spectrograms = merge_training_set(validation_samples)

	video_normalizer = data_processor.VideoNormalizer(train_video_samples)
	video_normalizer.normalize(train_video_samples)
	video_normalizer.normalize(validation_video_samples)

	with open(args.normalization_cache, 'wb') as normalization_fd:
		pickle.dump(video_normalizer, normalization_fd)

	network = SpeechEnhancementNetwork.build(train_mixed_spectrograms.shape[1:], train_video_samples.shape[1:])
	network.train(
		train_mixed_spectrograms, train_video_samples, train_speech_spectrograms,
		validation_mixed_spectrograms, validation_video_samples, validation_speech_spectrograms,
		args.model_cache_dir, args.tensorboard_dir
	)

	network.save(args.model_cache_dir)


def predict(args):
	storage = PredictionStorage(args.prediction_output_dir)
	network = SpeechEnhancementNetwork.load(args.model_cache_dir)

	with open(args.normalization_cache, 'rb') as normalization_fd:
		video_normalizer = pickle.load(normalization_fd)

	samples = load_preprocessed_samples(args.preprocessed_blob_paths)
	for sample in samples:
		try:
			print("predicting (%s, %s)..." % (sample.video_file_path, sample.noise_file_path))

			video_normalizer.normalize(sample.video_samples)

			loss = network.evaluate(sample.mixed_spectrograms, sample.video_samples, sample.speech_spectrograms)
			print("loss: %f" % loss)

			predicted_speech_spectrograms = network.predict(sample.mixed_spectrograms, sample.video_samples)

			predicted_speech_signal = data_processor.reconstruct_speech_signal(
				sample.mixed_signal, predicted_speech_spectrograms, sample.video_frame_rate, sample.peak
			)

			storage.save_prediction(sample, predicted_speech_signal)

		except Exception:
			logging.exception("failed to predict %s. skipping" % sample.video_file_path)


class PredictionStorage(object):

	def __init__(self, storage_dir):
		self.__base_dir = os.path.join(storage_dir, '{:%Y-%m-%d_%H-%M-%S}'.format(datetime.now()))
		os.mkdir(self.__base_dir)

	def __create_speaker_dir(self, speaker_id):
		speaker_dir = os.path.join(self.__base_dir, speaker_id)

		if not os.path.exists(speaker_dir):
			os.mkdir(speaker_dir)

		return speaker_dir

	def save_prediction(self, sample, predicted_speech_signal):
		speaker_dir = self.__create_speaker_dir(sample.speaker_id)

		speech_name = os.path.splitext(os.path.basename(sample.video_file_path))[0]
		noise_name = os.path.splitext(os.path.basename(sample.noise_file_path))[0]

		sample_prediction_dir = os.path.join(speaker_dir, speech_name + "_" + noise_name)
		os.mkdir(sample_prediction_dir)

		mixture_audio_path = os.path.join(sample_prediction_dir, "mixture.wav")
		enhanced_speech_audio_path = os.path.join(sample_prediction_dir, "enhanced.wav")
		source_audio_path = os.path.join(sample_prediction_dir, "source.wav")
		noise_audio_path = os.path.join(sample_prediction_dir, "noise.wav")

		copy2(sample.speech_file_path, source_audio_path)
		copy2(sample.noise_file_path, noise_audio_path)

		sample.mixed_signal.save_to_wav_file(mixture_audio_path)
		predicted_speech_signal.save_to_wav_file(enhanced_speech_audio_path)

		video_extension = os.path.splitext(os.path.basename(sample.video_file_path))[1]
		mixture_video_path = os.path.join(sample_prediction_dir, "mixture" + video_extension)
		enhanced_speech_video_path = os.path.join(sample_prediction_dir, "enhanced" + video_extension)

		ffmpeg.merge(sample.video_file_path, mixture_audio_path, mixture_video_path)
		ffmpeg.merge(sample.video_file_path, enhanced_speech_audio_path, enhanced_speech_video_path)


def list_speakers(args):
	if args.speakers is None:
		dataset = AudioVisualDataset(args.dataset_dir)
		speaker_ids = dataset.list_speakers()
	else:
		speaker_ids = args.speakers

	if args.ignored_speakers is not None:
		for speaker_id in args.ignored_speakers:
			speaker_ids.remove(speaker_id)

	return speaker_ids


def list_data(dataset_dir, speaker_ids, noise_dirs, max_files=None, shuffle=True):
	speech_dataset = AudioVisualDataset(dataset_dir)
	speech_subset = speech_dataset.subset(speaker_ids, max_files, shuffle)

	noise_dataset = AudioDataset(noise_dirs)
	noise_file_paths = noise_dataset.subset(max_files, shuffle)

	n_files = min(len(speech_subset), len(noise_file_paths))

	return speech_subset[:n_files], noise_file_paths[:n_files]


def main():
	parser = argparse.ArgumentParser(add_help=False)
	action_parsers = parser.add_subparsers()

	preprocess_parser = action_parsers.add_parser("preprocess")
	preprocess_parser.add_argument("--dataset_dir", type=str, required=True)
	preprocess_parser.add_argument("--noise_dirs", nargs="+", type=str, required=True)
	preprocess_parser.add_argument("--preprocessed_blob_path", type=str, required=True)
	preprocess_parser.add_argument("--speakers", nargs="+", type=str)
	preprocess_parser.add_argument("--ignored_speakers", nargs="+", type=str)
	preprocess_parser.set_defaults(func=preprocess)

	train_parser = action_parsers.add_parser("train")
	train_parser.add_argument("--train_preprocessed_blob_paths", nargs="+", type=str, required=True)
	train_parser.add_argument("--validation_preprocessed_blob_paths", nargs="+", type=str, required=True)
	train_parser.add_argument("--normalization_cache", type=str, required=True)
	train_parser.add_argument("--model_cache_dir", type=str, required=True)
	train_parser.add_argument("--tensorboard_dir", type=str, required=True)
	train_parser.set_defaults(func=train)

	predict_parser = action_parsers.add_parser("predict")
	predict_parser.add_argument("--preprocessed_blob_paths", nargs="+", type=str, required=True)
	predict_parser.add_argument("--model_cache_dir", type=str, required=True)
	predict_parser.add_argument("--normalization_cache", type=str, required=True)
	predict_parser.add_argument("--prediction_output_dir", type=str, required=True)
	predict_parser.set_defaults(func=predict)

	args = parser.parse_args()
	args.func(args)


if __name__ == "__main__":
	main()
