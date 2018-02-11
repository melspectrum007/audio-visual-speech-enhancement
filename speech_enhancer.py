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

BASE_FOLDER = '/cs/labs/peleg/asaph/playground/audio-visual-speech-enhancement' # todo: remove before releasing code

def preprocess(args):
	dataset_path = os.path.join(args.base_folder, 'data', args.dataset)
	cache_dir = os.path.join(args.base_folder, 'cache')
	if not os.path.exists(cache_dir):
		os.mkdir(cache_dir)
	preprocessed_dir = os.path.join(cache_dir, 'preprocessed')
	if not os.path.exists(preprocessed_dir):
		os.mkdir(preprocessed_dir)

	preprocessed_blob_path = os.path.join(preprocessed_dir, args.data_name + '.npz')

	speaker_ids = list_speakers(args)

	video_file_paths, speech_file_paths, noise_file_paths = list_data(
		dataset_path, speaker_ids, args.noise_dirs, max_files=1500
	)

	video_samples, mixed_spectrograms, speech_spectrograms = utils.preprocess_data(
		video_file_paths, speech_file_paths, noise_file_paths
	)

	np.savez(
		preprocessed_blob_path,
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
	cache_dir = os.path.join(args.base_folder, 'cache')
	if not os.path.exists(cache_dir):
		os.mkdir(cache_dir)
	models_dir = os.path.join(cache_dir, 'models')
	if not os.path.exists(models_dir):
		os.mkdir(models_dir)
	model_cache_dir = os.path.join(models_dir, args.model)
	if not os.path.exists(model_cache_dir):
		os.mkdir(model_cache_dir)

	normalization_cache_path = os.path.join(model_cache_dir + 'normalization.pkl')
	train_preprocessed_blob_paths = [os.path.join(args.base_folder, 'cache/preprocessed', p + '.npz') for p in args.train_data_names]
	val_preprocessed_blob_paths = [os.path.join(args.base_folder, 'cache/preprocessed', p + '.npz') for p in args.val_data_names]

	train_video_samples, train_mixed_spectrograms, train_speech_spectrograms = load_preprocessed_samples(
		train_preprocessed_blob_paths, max_samples=None
	)

	validation_video_samples, validation_mixed_spectrograms, validation_speech_spectrograms = load_preprocessed_samples(
		val_preprocessed_blob_paths, max_samples=None
	)

	# for overfit only:
	train_mixed_spectrograms = train_mixed_spectrograms[:, :, :-1, :, :]
	train_speech_spectrograms = train_speech_spectrograms[:, :, :-1, :, :]
	validation_mixed_spectrograms = validation_mixed_spectrograms[:, :, :-1, :, :]
	validation_speech_spectrograms = validation_speech_spectrograms[:, :, :-1, :, :]

	train_mixed_spectrograms = train_mixed_spectrograms.reshape((-1,) + train_mixed_spectrograms.shape[2:])
	train_speech_spectrograms = train_speech_spectrograms.reshape((-1,) + train_speech_spectrograms.shape[2:])
	validation_mixed_spectrograms = validation_mixed_spectrograms.reshape((-1,) + validation_mixed_spectrograms.shape[2:])
	validation_speech_spectrograms = validation_speech_spectrograms.reshape((-1,) + validation_speech_spectrograms.shape[2:])


	train_video_samples = train_video_samples.reshape((-1,) + train_video_samples.shape[2:])
	validation_video_samples = validation_video_samples.reshape((-1,) + validation_video_samples.shape[2:])

	# train_mixed_spectrograms = train_mixed_spectrograms[:,:-1,:,:]
	# train_speech_spectrograms = train_speech_spectrograms[:,:-1,:,:]
	# validation_mixed_spectrograms = validation_mixed_spectrograms[:,:-1,:,:]
	# validation_speech_spectrograms = validation_speech_spectrograms[:,:-1,:,:]

	video_normalizer = dp.VideoNormalizer(train_video_samples)
	video_normalizer.normalize(train_video_samples)
	video_normalizer.normalize(validation_video_samples)

	with open(normalization_cache_path, 'wb') as normalization_fd:
		pickle.dump(video_normalizer, normalization_fd)

	network = SpeechEnhancementNetwork.build(train_mixed_spectrograms.shape[1:], train_video_samples.shape[1:])
	network.train(
		train_mixed_spectrograms, train_video_samples, train_speech_spectrograms,
		validation_mixed_spectrograms, validation_video_samples, validation_speech_spectrograms,
		model_cache_dir
	)

	network.save(model_cache_dir)

def test_on_set(args):
	cache_dir = os.path.join(args.base_folder, 'cache')
	prediction_output_dir = os.path.join(args.base_folder, 'out', args.model)
	if not os.path.exists(prediction_output_dir):
		os.mkdir(prediction_output_dir)
	if not os.path.exists(cache_dir):
		os.mkdir(cache_dir)
	models_dir = os.path.join(cache_dir, 'models')
	if not os.path.exists(models_dir):
		os.mkdir(models_dir)
	model_cache_dir = os.path.join(models_dir, args.model)
	if not os.path.exists(model_cache_dir):
		os.mkdir(model_cache_dir)

	normalization_cache_path = os.path.join(model_cache_dir + 'normalization.pkl')
	train_preprocessed_blob_path = os.path.join(args.base_folder, 'cache/preprocessed', args.data_name + '.npz')

	test_video_samples, test_mixed_spectrograms, test_speech_spectrograms = load_preprocessed_samples(
		[train_preprocessed_blob_path], max_samples=None
	)

	# for overfit only:
	test_mixed_spectrograms = test_mixed_spectrograms[:, :, :-1, :, :]
	orig_shape = test_mixed_spectrograms.shape

	test_mixed_spectrograms = test_mixed_spectrograms.reshape((-1,) + test_mixed_spectrograms.shape[2:])

	test_video_samples = test_video_samples.reshape((-1,) + test_video_samples.shape[2:])

	video_normalizer = dp.VideoNormalizer(test_video_samples)
	video_normalizer.normalize(test_video_samples)

	with open(normalization_cache_path, 'wb') as normalization_fd:
		pickle.dump(video_normalizer, normalization_fd)

	network = SpeechEnhancementNetwork.load(model_cache_dir)
	enhanced_stft = network.predict(test_mixed_spectrograms, test_video_samples)

	enhanced_stft = enhanced_stft.reshape(orig_shape)

	data_processor = utils.DataProcessor(25, 16000)
	mixed_for_norm = AudioSignal.from_wav_file(BASE_FOLDER + '/mixture.wav')
	data_processor.mean, data_processor.std = mixed_for_norm.normalize()

	# storage = PredictionStorage(prediction_output_dir)
	for	i in range(enhanced_stft.shape[0]):
		enhanced = np.concatenate(list(enhanced_stft[i]), 1)
		signal = data_processor.reconstruct_signal(enhanced, 16000)
		# mixed = data_processor.reconstruct_signal(test_mixed_spectrograms, 16000)
		signal.save_to_wav_file(os.path.join(prediction_output_dir, 'enhanced_' + str(i) + '.wav'))
		# mixed.save_to_wav_file(os.path.join(prediction_output_dir, 'mixed_' + str(i) + '.wav'))



def predict(args):
	model_cache_dir = os.path.join(args.base_folder, 'cache/models', args.model)
	prediction_output_dir = os.path.join(args.base_folder, 'out', args.model)
	normalization_cache = os.path.join(model_cache_dir, 'normalization.pkl')
	dataset_path = os.path.join(args.base_folder, 'data', args.dataset, 'test')
	if not os.path.exists(prediction_output_dir):
		os.mkdir(prediction_output_dir)

	storage = PredictionStorage(args.prediction_output_dir)
	network = SpeechEnhancementNetwork.load(args.model_cache_dir)
	network.summerize()

	with open(normalization_cache, 'rb') as normalization_fd:
		video_normalizer = pickle.load(normalization_fd)

	speaker_ids = list_speakers(args)
	for speaker_id in speaker_ids:
		video_file_paths, speech_file_paths, noise_file_paths = list_data(
			dataset_path, [speaker_id], args.noise_dirs, max_files=5, shuffle=False
		)

		fps = VideoFileReader(video_file_paths[0]).get_frame_rate()
		sr = AudioSignal.from_wav_file(speech_file_paths[0]).get_sample_rate()

		data_processor = utils.DataProcessor(fps, sr)

		for video_file_path, speech_file_path, noise_file_path in zip(video_file_paths, speech_file_paths, noise_file_paths):
			try:
				print('predicting (%s, %s)...' % (video_file_path, noise_file_path))
				mixed_signal = utils.mix_source_noise(speech_file_path, noise_file_path)
				video_samples, mixed_stfts, label_stfts = data_processor.preprocess_sample(
					video_file_path, speech_file_path, noise_file_path)
				video_normalizer.normalize(video_samples)

				spec_dict = {}

				mixed_stfts = mixed_stfts[:, :-1, :, :]

				enhanced_stft = network.predict(mixed_stfts, video_samples)
				enhanced_stft = np.concatenate(list(enhanced_stft), axis=1)
				label_stft = np.concatenate(list(label_stfts), axis=1)
				mixed_stft = data_processor.get_stft(mixed_signal.get_data())

				spec_dict['mixed real'] = mixed_stft[:,:,0]
				spec_dict['mixed imag'] = mixed_stft[:,:,1]
				spec_dict['label real'] = label_stft[:,:,0]
				spec_dict['label imag'] = label_stft[:,:,1]
				spec_dict['enhanced real'] = np.sign(enhanced_stft[:,:,0]) * np.log(np.abs(enhanced_stft[:,:,0]) + 1)
				spec_dict['enhanced imag'] = np.sign(enhanced_stft[:,:,1]) * np.log(np.abs(enhanced_stft[:,:,1]) + 1)

				predicted_speech_signal = data_processor.reconstruct_signal(enhanced_stft, mixed_signal)

				sample_dir = storage.save_prediction(
					speaker_id, video_file_path, noise_file_path, speech_file_path,
					mixed_signal, predicted_speech_signal, enhanced_stft
				)

				save_spectrograms(spec_dict, sample_dir)

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

def save_spectrograms(spec_dict, dir_path):
	for name, spec in spec_dict.iteritems():
		np.save(os.path.join(dir_path, name), spec)

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

	parser.add_argument('-bf', '--base_folder', type=str, default=BASE_FOLDER)

	action_parsers = parser.add_subparsers()

	preprocess_parser = action_parsers.add_parser('preprocess')
	preprocess_parser.add_argument('-dn', '--data_name', type=str, required=True)
	preprocess_parser.add_argument('-ds', '--dataset', type=str, required=True)
	preprocess_parser.add_argument('-s', '--speakers', nargs='+', type=str)
	preprocess_parser.add_argument('-is', '--ignored_speakers', nargs='+', type=str)
	preprocess_parser.add_argument('-n', '--noise_dirs', nargs='+', type=str, required=True)
	preprocess_parser.set_defaults(func=preprocess, which='preprocess')

	# generate_parser = action_parsers.add_parser('generate_vocoder_dataset')
	# generate_parser.add_argument('-tdn', '--train_data_name', type=str, required=True)
	# generate_parser.add_argument('-mn', '--model', type=str, required=True)
	# generate_parser.add_argument('-fps', '--frames_per_second', type=int, default=25)
	# generate_parser.add_argument('-sr', '--sampling_rate', type=int, default=16000)
	# generate_parser.set_defaults(func=generate_vocoder_dataset)

	train_parser = action_parsers.add_parser('train')
	train_parser.add_argument('-mn', '--model', type=str, required=True)
	train_parser.add_argument('-tdn', '--train_data_names', nargs='+', type=str, required=True)
	train_parser.add_argument('-vdn', '--val_data_names', nargs='+', type=str, required=True)
	train_parser.set_defaults(func=train)

	# train_vocoder_parser = action_parsers.add_parser('train_vocoder')
	# train_vocoder_parser.add_argument('-mn', '--model', type=str, required=True)
	# train_vocoder_parser.add_argument('-tdn', '--train_data_name', type=str, required=True)
	# train_vocoder_parser.add_argument('-vdn', '--val_data_name', type=str, required=True)
	# train_vocoder_parser.set_defaults(func=train_vocoder)

	predict_parser = action_parsers.add_parser('predict')
	predict_parser.add_argument('-mn', '--model', type=str, required=True)
	predict_parser.add_argument('-ds', '--dataset', type=str, required=True)
	predict_parser.add_argument('-s', '--speakers', nargs='+', type=str, required=True)
	predict_parser.add_argument('-is', '--ignored_speakers', nargs='+', type=str)
	predict_parser.add_argument('-n', '--noise_dirs', nargs='+', type=str, required=True)
	predict_parser.set_defaults(func=predict)

	test_on_set_parser = action_parsers.add_parser('test_on_set')
	test_on_set_parser.add_argument('-mn', '--model', type=str, required=True)
	test_on_set_parser.add_argument('-dn', '--data_name', type=str, required=True)
	test_on_set_parser.set_defaults(func=test_on_set)

	# predict_vocoder_parser = action_parsers.add_parser('predict_vocoder')
	# predict_vocoder_parser.add_argument('-mn', '--model', type=str, required=True)
	# predict_vocoder_parser.add_argument('-dn', '--data_name', type=str, required=True)
	# predict_vocoder_parser.set_defaults(func=predict_vocoder)
	#
	# test_parser = action_parsers.add_parser('test')
	# test_parser.add_argument('-mn', '--model', type=str, required=True)
	# test_parser.add_argument('-p', '--paths', type=str, nargs='+', required=True)
	# test_parser.set_defaults(func=test, which='test')

	args = parser.parse_args()
	args.func(args)


if __name__ == '__main__':
	main()
