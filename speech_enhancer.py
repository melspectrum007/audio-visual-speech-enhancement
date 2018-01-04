import argparse, os, logging, pickle
import numpy as np
import librosa as lb
import utils
import data_processor as dp

from dataset import AudioVisualDataset, AudioDataset
from network import SpeechEnhancementNetwork
from shutil import copy2
from mediaio import ffmpeg
from datetime import datetime
from mediaio.video_io import VideoFileReader
from mediaio.audio_io import AudioSignal

def preprocess(args):
	speaker_ids = list_speakers(args)

	video_file_paths, speech_file_paths, noise_file_paths = list_data(
		args.dataset_dir, speaker_ids, args.noise_dirs, max_files=1500
	)

	video_samples, mixed_spectrograms, speech_spectrograms = utils.preprocess_data(
		video_file_paths, speech_file_paths, noise_file_paths
	)

	np.savez(
		args.preprocessed_blob_path,
		video_samples=video_samples,
		mixed_spectrograms=mixed_spectrograms,
		speech_spectrograms=speech_spectrograms,
	)


def load_preprocessed_samples(preprocessed_blob_paths, max_samples=None):
	all_video_samples = []
	all_mixed_spectrograms = []
	all_speech_spectrograms = []

	for preprocessed_blob_path in preprocessed_blob_paths:
		print('loading preprocessed samples from %s' % preprocessed_blob_path)
		
		with np.load(preprocessed_blob_path) as data:
			all_video_samples.append(data['video_samples'][:max_samples])
			all_mixed_spectrograms.append(data['mixed_spectrograms'][:max_samples])
			all_speech_spectrograms.append(data['speech_spectrograms'][:max_samples])

	video_samples = np.concatenate(all_video_samples, axis=0)
	mixed_spectrograms = np.concatenate(all_mixed_spectrograms, axis=0)
	speech_spectrograms = np.concatenate(all_speech_spectrograms, axis=0)

	permutation = np.random.permutation(video_samples.shape[0])
	video_samples = video_samples[permutation]
	mixed_spectrograms = mixed_spectrograms[permutation]
	speech_spectrograms = speech_spectrograms[permutation]

	return (
		video_samples,
		mixed_spectrograms,
		speech_spectrograms,
	)


def train(args):
	train_video_samples, train_mixed_spectrograms, train_speech_spectrograms = load_preprocessed_samples(
		args.train_preprocessed_blob_paths, max_samples=None
	)

	validation_video_samples, validation_mixed_spectrograms, validation_speech_spectrograms = load_preprocessed_samples(
		args.validation_preprocessed_blob_paths, max_samples=None
	)

	train_mixed_spectrograms = train_mixed_spectrograms[:,:-1,:,:]
	train_speech_spectrograms = train_speech_spectrograms[:,:-1,:,:]
	validation_mixed_spectrograms = validation_mixed_spectrograms[:,:-1,:,:]
	validation_speech_spectrograms = validation_speech_spectrograms[:,:-1,:,:]

	video_normalizer = dp.VideoNormalizer(train_video_samples)
	video_normalizer.normalize(train_video_samples)
	video_normalizer.normalize(validation_video_samples)

	with open(args.normalization_cache, 'wb') as normalization_fd:
		pickle.dump(video_normalizer, normalization_fd)

	network = SpeechEnhancementNetwork.build(train_mixed_spectrograms.shape[1:-1], train_video_samples.shape[1:])
	network.train(
		train_mixed_spectrograms, train_video_samples, train_speech_spectrograms,
		validation_mixed_spectrograms, validation_video_samples, validation_speech_spectrograms,
		args.model_cache_dir, args.tensorboard_dir
	)

	network.save(args.model_cache_dir)


def predict(args):
	storage = PredictionStorage(args.prediction_output_dir)
	network = SpeechEnhancementNetwork.load(args.model_cache_dir)
	network.__model.summary()

	with open(args.normalization_cache, 'rb') as normalization_fd:
		video_normalizer = pickle.load(normalization_fd)

	speaker_ids = list_speakers(args)
	for speaker_id in speaker_ids:
		video_file_paths, speech_file_paths, noise_file_paths = list_data(
			args.dataset_dir, [speaker_id], args.noise_dirs, max_files=2, shuffle=True
		)

		fps = VideoFileReader(video_file_paths[0]).get_frame_rate()
		sr = AudioSignal.from_wav_file(speech_file_paths[0]).get_sample_rate()

		data_processor = utils.DataProcessor(fps, sr)

		for video_file_path, speech_file_path, noise_file_path in zip(video_file_paths, speech_file_paths, noise_file_paths):
			try:
				print('predicting (%s, %s)...' % (video_file_path, noise_file_path))
				mixed_signal = utils.mix_source_noise(speech_file_path, noise_file_path)
				video_samples, mixed_spectrograms, label_stfts = data_processor.preprocess_sample(
					video_file_path, speech_file_path, noise_file_path)

				video_normalizer.normalize(video_samples)

				# loss = network.evaluate(mixed_spectrograms, video_samples, speech_spectrograms)
				# print('loss: %f' % loss)
				mixed_spectrograms = mixed_spectrograms[:, :-1, :, :]
				enhanced_real, enhanced_imag = network.predict(mixed_spectrograms, video_samples)

				enhanced_real = np.concatenate(list(enhanced_real), axis=1)
				enhanced_imag = np.concatenate(list(enhanced_imag), axis=1)

				mixed_stft = data_processor.get_stft(mixed_signal.get_data())
				mixed_real = mixed_stft[:,:, 0]
				mixed_imag = mixed_stft[:,:, 1]

				label_real = np.concatenate(list(label_stfts[:,:,:,0]), axis=1)
				label_imag = np.concatenate(list(label_stfts[:,:,:,1]), axis=1)

				predicted_speech_signal = data_processor.reconstruct_signal(enhanced_real, enhanced_imag, mixed_signal)

				sample_dir = storage.save_prediction(
					speaker_id, video_file_path, noise_file_path, speech_file_path,
					mixed_signal, predicted_speech_signal, enhanced_real
				)

				storage.save_spectrograms([enhanced_real, enhanced_imag, mixed_real, mixed_imag, label_real, label_imag],
										  ['enhanced real', 'endanced imag', 'mixed real', 'mixed imag', 'source real',  'source imag'],
										  sample_dir)

			except Exception:
				logging.exception('failed to predict %s. skipping' % video_file_path)


class PredictionStorage(object):

	def __init__(self, storage_dir):
		self.__base_dir = os.path.join(storage_dir, '{:%Y-%m-%d_%H-%M-%S}'.format(datetime.now()))
		os.mkdir(self.__base_dir)

	def __create_speaker_dir(self, speaker_id):
		speaker_dir = os.path.join(self.__base_dir, speaker_id)

		if not os.path.exists(speaker_dir):
			os.mkdir(speaker_dir)

		return speaker_dir

	def save_prediction(self, speaker_id, video_file_path, noise_file_path, speech_file_path,
						mixed_signal, predicted_speech_signal, speech_spec):

		speaker_dir = self.__create_speaker_dir(speaker_id)

		speech_name = os.path.splitext(os.path.basename(video_file_path))[0]
		noise_name = os.path.splitext(os.path.basename(noise_file_path))[0]

		sample_prediction_dir = os.path.join(speaker_dir, speech_name + '_' + noise_name)
		os.mkdir(sample_prediction_dir)

		mixture_audio_path = os.path.join(sample_prediction_dir, 'mixture.wav')
		enhanced_speech_audio_path = os.path.join(sample_prediction_dir, 'enhanced.wav')
		source_speech_new_audio_path = os.path.join(sample_prediction_dir, 'source.wav')
		copy2(speech_file_path, source_speech_new_audio_path)

		mixed_signal.save_to_wav_file(mixture_audio_path)
		predicted_speech_signal.save_to_wav_file(enhanced_speech_audio_path)

		video_extension = os.path.splitext(os.path.basename(video_file_path))[1]
		mixture_video_path = os.path.join(sample_prediction_dir, 'mixture' + video_extension)
		enhanced_speech_video_path = os.path.join(sample_prediction_dir, 'enhanced' + video_extension)

		ffmpeg.merge(video_file_path, mixture_audio_path, mixture_video_path)
		ffmpeg.merge(video_file_path, enhanced_speech_audio_path, enhanced_speech_video_path)

		# os.unlink(mixture_audio_path)
		# os.unlink(enhanced_speech_audio_path)

		return sample_prediction_dir

	def save_spectrograms(self, spectrograms, names, dir_path):
		for i, spec in enumerate(spectrograms):
			np.save(os.path.join(dir_path, names[i]), spec)

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
	speech_subset = speech_dataset.subset(speaker_ids, max_files, shuffle=shuffle)

	noise_dataset = AudioDataset(noise_dirs)
	noise_file_paths = noise_dataset.subset(max_files, shuffle=shuffle)

	n_files = min(speech_subset.size(), len(noise_file_paths))

	return speech_subset.video_paths()[:n_files], speech_subset.audio_paths()[:n_files], noise_file_paths[:n_files]


def main():
	parser = argparse.ArgumentParser(add_help=False)
	action_parsers = parser.add_subparsers()

	preprocess_parser = action_parsers.add_parser('preprocess')
	preprocess_parser.add_argument('--dataset_dir', type=str, required=True)
	preprocess_parser.add_argument('--noise_dirs', nargs='+', type=str, required=True)
	preprocess_parser.add_argument('--preprocessed_blob_path', type=str, required=True)
	preprocess_parser.add_argument('--speakers', nargs='+', type=str)
	preprocess_parser.add_argument('--ignored_speakers', nargs='+', type=str)
	preprocess_parser.set_defaults(func=preprocess)

	train_parser = action_parsers.add_parser('train')
	train_parser.add_argument('--train_preprocessed_blob_paths', nargs='+', type=str, required=True)
	train_parser.add_argument('--validation_preprocessed_blob_paths', nargs='+', type=str, required=True)
	train_parser.add_argument('--normalization_cache', type=str, required=True)
	train_parser.add_argument('--model_cache_dir', type=str, required=True)
	train_parser.add_argument('--tensorboard_dir', type=str, required=False)
	train_parser.set_defaults(func=train)

	predict_parser = action_parsers.add_parser('predict')
	predict_parser.add_argument('--dataset_dir', type=str, required=True)
	predict_parser.add_argument('--noise_dirs', nargs='+', type=str, required=True)
	predict_parser.add_argument('--model_cache_dir', type=str, required=True)
	predict_parser.add_argument('--normalization_cache', type=str, required=True)
	predict_parser.add_argument('--prediction_output_dir', type=str, required=True)
	predict_parser.add_argument('--speakers', nargs='+', type=str)
	predict_parser.add_argument('--ignored_speakers', nargs='+', type=str)
	predict_parser.set_defaults(func=predict)

	args = parser.parse_args()
	args.func(args)


if __name__ == '__main__':
	main()
