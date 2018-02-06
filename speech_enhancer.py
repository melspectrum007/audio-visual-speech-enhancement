
import argparse, os, logging, pickle
import numpy as np
import utils
import data_processor as dp

from dataset import AudioVisualDataset, AudioDataset
from network import SpeechEnhancementNetwork
from wavenet_vocoder import WavenetVocoder
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

	video_file_paths, source_file_paths, noise_file_paths = list_data(
		dataset_path, speaker_ids, args.noise_dirs, max_files=1500
	)

	video_samples, mixed_spectrograms, source_spectrograms, source_phases = utils.preprocess_data(
		video_file_paths, source_file_paths, noise_file_paths
	)

	np.savez(
		preprocessed_blob_path,
		video_samples=video_samples,
		mixed_spectrograms=mixed_spectrograms,
		source_spectrograms=source_spectrograms,
		source_phases=source_phases,
	)


def load_preprocessed_samples(preprocessed_blob_paths, max_samples=None):
	all_video_samples = []
	all_mixed_spectrograms = []
	all_source_spectrograms = []
	all_source_phases = []

	for preprocessed_blob_path in preprocessed_blob_paths:
		print('loading preprocessed samples from %s' % preprocessed_blob_path)
		
		with np.load(preprocessed_blob_path) as data:
			all_video_samples.append(data['video_samples'][:max_samples])
			all_mixed_spectrograms.append(data['mixed_spectrograms'][:max_samples])
			all_source_spectrograms.append(data['source_spectrograms'][:max_samples])
			all_source_phases.append(data['source_phases'][:max_samples])

	video_samples = np.concatenate(all_video_samples, axis=0)
	mixed_spectrograms = np.concatenate(all_mixed_spectrograms, axis=0)
	source_spectrograms = np.concatenate(all_source_spectrograms, axis=0)
	source_phases = np.concatenate(all_source_phases, axis=0)


	permutation = np.random.permutation(video_samples.shape[0])
	video_samples = video_samples[permutation]
	mixed_spectrograms = mixed_spectrograms[permutation]
	source_spectrograms = source_spectrograms[permutation]
	source_phases = source_phases[permutation]

	return (
		video_samples,
		mixed_spectrograms,
		source_spectrograms,
		source_phases,
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

	train_video_samples, train_mixed_spectrograms, train_source_spectrograms = load_preprocessed_samples(
		train_preprocessed_blob_paths, max_samples=None
	)[:3]

	height, width, frames = train_video_samples.shape[2:]
	freq, bins = train_mixed_spectrograms.shape[2:]

	train_video_samples = train_video_samples.reshape(-1, height, width, frames)
	train_mixed_spectrograms = train_mixed_spectrograms.reshape(-1, freq, bins)
	train_source_spectrograms = train_source_spectrograms.reshape(-1, freq, bins)

	val_video_samples, val_mixed_spectrograms, val_source_spectrograms = load_preprocessed_samples(
		val_preprocessed_blob_paths, max_samples=None
	)[:3]

	val_video_samples = val_video_samples.reshape(-1, height, width, frames)
	val_mixed_spectrograms = val_mixed_spectrograms.reshape(-1, freq, bins)
	val_source_spectrograms = val_source_spectrograms.reshape(-1, freq, bins)

	print 'normalizing video samples...'
	video_normalizer = dp.VideoNormalizer(train_video_samples)
	video_normalizer.normalize(train_video_samples)
	video_normalizer.normalize(val_video_samples)

	with open(normalization_cache_path, 'wb') as normalization_fd:
		pickle.dump(video_normalizer, normalization_fd)

	print 'building network...'
	network = SpeechEnhancementNetwork.build(train_mixed_spectrograms.shape[1:], train_video_samples.shape[1:])
	network.train(
		train_mixed_spectrograms, train_video_samples, train_source_spectrograms,
		val_mixed_spectrograms, val_video_samples, val_source_spectrograms,
		model_cache_dir
	)

	network.save(args.model_cache_dir)


def predict(args):
	model_cache_dir = os.path.join(args.base_folder, 'cache/models', args.model)
	prediction_output_dir = os.path.join(args.base_folder, 'out', args.model)
	normalization_cache = os.path.join(model_cache_dir, 'normalization.pkl')
	dataset_path = os.path.join(args.base_folder, 'data', args.dataset, 'test')
	if not os.path.exists(prediction_output_dir):
		os.mkdir(prediction_output_dir)

	storage = PredictionStorage(prediction_output_dir)
	network = SpeechEnhancementNetwork.load(model_cache_dir)

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
				video_samples, mixed_spectrograms, label_spectrograms = data_processor.preprocess_sample(
					video_file_path, speech_file_path, noise_file_path)[:3]

				video_normalizer.normalize(video_samples)

				# loss = network.evaluate(mixed_spectrograms, video_samples, speech_spectrograms)
				# print('loss: %f' % loss)
				enhanced_speech_spectrograms = network.predict(mixed_spectrograms, video_samples)

				enhanced_spec = np.concatenate(list(enhanced_speech_spectrograms), axis=1)
				mixed_spec = data_processor.get_mag_phase(mixed_signal.get_data())[0]
				label_spec = np.concatenate(list(label_spectrograms), axis=1)

				predicted_speech_signal = data_processor.reconstruct_signal(enhanced_spec, mixed_signal, use_griffin_lim=False)

				sample_dir = storage.save_prediction(
					speaker_id, video_file_path, noise_file_path, speech_file_path,
					mixed_signal, predicted_speech_signal, enhanced_spec
				)

				storage.save_spectrograms([enhanced_spec, mixed_spec, label_spec], ['enhanced', 'mixed', 'source'], sample_dir)

			except Exception:
				logging.exception('failed to predict %s. skipping' % video_file_path)


def test(args):
	model_cache_dir = os.path.join(args.base_folder, 'cache/models', args.model)
	normalization_path = os.path.join(model_cache_dir, 'normalization.pkl')

	network = SpeechEnhancementNetwork.load(model_cache_dir)
	with open(normalization_path, 'rb') as normalization_fd:
		video_normalizer = pickle.load(normalization_fd)

	input_paths = args.paths

	for input_path in input_paths:
		with VideoFileReader(input_path) as reader:
			fps = reader.get_frame_rate()
			ffmpeg.extract_audio(input_path, '/cs/grad/asaph/testing/tmp.wav')
			mixed_signal = AudioSignal.from_wav_file('/cs/grad/asaph/testing/tmp.wav')
			sr = mixed_signal.get_sample_rate()

			dataProcessor = utils.DataProcessor(fps, sr)
			video_sampels, mixed_spectrograms = dataProcessor.preprocess_inputs(reader.read_all_frames(convert_to_gray_scale=True), mixed_signal)
			video_normalizer.normalize(video_sampels)

			enhanced_speech_spectrograms = network.predict(mixed_spectrograms, video_sampels)
			enhanced_spec = np.concatenate(list(enhanced_speech_spectrograms), axis=1)
			mixed_spec = np.concatenate(list(mixed_spectrograms), axis=1)

			# mixed_signal = AudioSignal.from_wav_file('/cs/grad/asaph/testing/mixture.wav')

			predicted_speech_signal = dataProcessor.reconstruct_signal(enhanced_spec, mixed_signal)
			predicted_speech_signal.save_to_wav_file(os.path.splitext(input_path)[0] + '.wav')

			np.save(os.path.split(input_path)[0] + '/mixed.npy', mixed_spec)
			np.save(os.path.split(input_path)[0] + '/enhanced.npy', enhanced_spec)


def generate_vocoder_dataset(args):
	model_cache_dir = os.path.join(args.base_folder, 'cache/models', args.model)
	normalization_path = os.path.join(model_cache_dir, 'normalization.pkl')
	train_preprocessed_blob_paths = os.path.join(args.base_folder, 'cache/preprocessed', args.train_data_name + '.npz')
	vocoder_train_blob_path = os.path.join(args.base_folder, 'cache/preprocessed', args.train_data_name + '-vocoder-' + args.model + '.npz')

	network = SpeechEnhancementNetwork.load(model_cache_dir)
	with open(normalization_path, 'rb') as normalization_fd:
		video_normalizer = pickle.load(normalization_fd)

	dataProcessor = utils.DataProcessor(args.frames_per_second, args.sampling_rate)

	train_video_samples, train_mixed_spectrograms, train_source_spectrograms, train_source_phases = load_preprocessed_samples(
		[train_preprocessed_blob_paths], max_samples=None
	)

	vocoder_train_enhanced_spectrograms = []
	vocoder_train_source_waveforms = []
	for i in range(train_video_samples.shape[0]):
		video_samples = train_video_samples[i]
		video_normalizer.normalize(video_samples)
		mixed_spectrograms = train_mixed_spectrograms[i]

		enhanced_spectrograms = network.predict(mixed_spectrograms, video_samples)
		enhanced_spectrogram = np.concatenate(list(enhanced_spectrograms), axis=1)

		source_spectrograms = train_source_spectrograms[i]
		source_phase = train_source_phases[i]

		source_spectrogram = np.concatenate(list(source_spectrograms), axis=1)
		source_spectrogram = dataProcessor.recover_linear_spectrogram(source_spectrogram)

		waveform = dataProcessor.reconstruct_waveform_data(source_spectrogram, source_phase)

		vocoder_train_enhanced_spectrograms.append(enhanced_spectrogram)
		vocoder_train_source_waveforms.append(waveform)

	vocoder_train_enhanced_spectrograms = np.stack(vocoder_train_enhanced_spectrograms)
	vocoder_train_source_waveforms = np.stack(vocoder_train_source_waveforms)

	np.savez(
		vocoder_train_blob_path,
		enhanced_spectrograms=vocoder_train_enhanced_spectrograms,
		source_waveforms=vocoder_train_source_waveforms,
	)


def train_vocoder(args):
	cache_dir = os.path.join(args.base_folder, 'cache')
	if not os.path.exists(cache_dir):
		os.mkdir(cache_dir)
	models_dir = os.path.join(cache_dir, 'models')
	if not os.path.exists(models_dir):
		os.mkdir(models_dir)
	model_cache_dir = os.path.join(models_dir, args.model)
	if not os.path.exists(model_cache_dir):
		os.mkdir(model_cache_dir)

	train_preprocessed_blob_path = os.path.join(args.base_folder, 'cache/preprocessed', args.train_data_name + '.npz')
	val_preprocessed_blob_path = os.path.join(args.base_folder, 'cache/preprocessed', args.val_data_name + '.npz')

	with np.load(train_preprocessed_blob_path) as data:
		OVERFIT_NUM = 15 # todo: remove
		train_enhanced_spectrograms = data['enhanced_spectrograms'][:OVERFIT_NUM]
		train_waveforms = data['source_waveforms'][:OVERFIT_NUM]

	with np.load(val_preprocessed_blob_path) as data:
		val_enhanced_spectrograms = data['enhanced_spectrograms']
		val_waveforms = data['source_waveforms']

	val_enhanced_spectrograms = train_enhanced_spectrograms
	val_waveforms = train_waveforms

	print 'building network...'
	network = WavenetVocoder(80, 15, (train_enhanced_spectrograms.shape[1:]))
	network.train(train_enhanced_spectrograms, train_waveforms, val_enhanced_spectrograms, val_waveforms, model_cache_dir)


def predict_vocoder(args):
	model_cache_dir = os.path.join(args.base_folder, 'cache/models', args.model)
	prediction_output_dir = os.path.join(args.base_folder, 'out', args.model)
	testset_path = os.path.join(args.base_folder, 'cache/preprocessed', args.data_name + '.npz')
	if not os.path.exists(prediction_output_dir):
		os.mkdir(prediction_output_dir)

	storage = PredictionStorage(prediction_output_dir)
	network = WavenetVocoder.load(model_cache_dir)

	with np.load(testset_path) as data:
		OVERFIT_NUM = 15 # todo: remove
		enhanced_spectrogarms = data['enhanced_spectrograms'][:OVERFIT_NUM]
		source_waveforms = data['source_waveforms'][:OVERFIT_NUM]

	for i in range(enhanced_spectrogarms.shape[0]):
		wave_data = network.predict_one_sample(enhanced_spectrogarms[i][np.newaxis, ...])
		enhanced_signal = AudioSignal(wave_data, 16000)
		source_signal = AudioSignal(source_waveforms[i], 16000)

		enhanced_signal.set_sample_type('int16')
		source_signal.set_sample_type('int16')

		storage.save_vocoder_pred(enhanced_signal, source_signal, i)


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

	def save_vocoder_pred(self, enhanced_signal, source_signal, num):
		enhanced_path = os.path.join(self.__base_dir, 'enhanced_' + str(num) + '.wav')
		source_path = os.path.join(self.__base_dir, 'source_' + str(num) + '.wav')
		enhanced_signal.save_to_wav_file(enhanced_path)
		source_signal.save_to_wav_file(source_path)


	def save_spectrograms(self, spectrograms, names, dir_path):
		for i, spec in enumerate(spectrograms):
			np.save(os.path.join(dir_path, names[i]), spec)


def list_speakers(args):
	if args.speakers is None:
		dataset = AudioVisualDataset(args.dataset)
		speaker_ids = dataset.list_speakers()
	else:
		speaker_ids = args.speakers

	if args.ignored_speakers is not None:
		for speaker_id in args.ignored_speakers:
			speaker_ids.remove(speaker_id)

	return speaker_ids


def list_data(dataset, speaker_ids, noise_dirs, max_files=None, shuffle=True):
	speech_dataset = AudioVisualDataset(dataset)
	speech_subset = speech_dataset.subset(speaker_ids, max_files, shuffle=shuffle)

	noise_dataset = AudioDataset(noise_dirs)
	noise_file_paths = noise_dataset.subset(max_files, shuffle=shuffle)[::-1]

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

	generate_parser = action_parsers.add_parser('generate_vocoder_dataset')
	generate_parser.add_argument('-tdn', '--train_data_name', type=str, required=True)
	generate_parser.add_argument('-mn', '--model', type=str, required=True)
	generate_parser.add_argument('-fps', '--frames_per_second', type=int, default=25)
	generate_parser.add_argument('-sr', '--sampling_rate', type=int, default=16000)
	generate_parser.set_defaults(func=generate_vocoder_dataset)

	train_parser = action_parsers.add_parser('train')
	train_parser.add_argument('-mn', '--model', type=str, required=True)
	train_parser.add_argument('-tdn', '--train_data_names', nargs='+', type=str, required=True)
	train_parser.add_argument('-vdn', '--val_data_names', nargs='+', type=str, required=True)
	train_parser.set_defaults(func=train)

	train_vocoder_parser = action_parsers.add_parser('train_vocoder')
	train_vocoder_parser.add_argument('-mn', '--model', type=str, required=True)
	train_vocoder_parser.add_argument('-tdn', '--train_data_name', type=str, required=True)
	train_vocoder_parser.add_argument('-vdn', '--val_data_name', type=str, required=True)
	train_vocoder_parser.set_defaults(func=train_vocoder)

	predict_parser = action_parsers.add_parser('predict')
	predict_parser.add_argument('-mn', '--model', type=str, required=True)
	predict_parser.add_argument('-ds', '--dataset', type=str, required=True)
	predict_parser.add_argument('-s', '--speakers', nargs='+', type=str, required=True)
	predict_parser.add_argument('-is', '--ignored_speakers', nargs='+', type=str)
	predict_parser.add_argument('-n', '--noise_dirs', nargs='+', type=str, required=True)
	predict_parser.set_defaults(func=predict)

	predict_vocoder_parser = action_parsers.add_parser('predict_vocoder')
	predict_vocoder_parser.add_argument('-mn', '--model', type=str, required=True)
	predict_vocoder_parser.add_argument('-dn', '--data_name', type=str, required=True)
	predict_vocoder_parser.set_defaults(func=predict_vocoder)

	test_parser = action_parsers.add_parser('test')
	test_parser.add_argument('-mn', '--model', type=str, required=True)
	test_parser.add_argument('-p', '--paths', type=str, nargs='+', required=True)
	test_parser.set_defaults(func=test, which='test')

	args = parser.parse_args()
	args.func(args)


if __name__ == '__main__':
	main()
