
import argparse, os, logging, pickle
import numpy as np
import utils

from dataset import AudioVisualDataset, AudioDataset
from network import SpeechEnhancementNetwork
from wavenet_vocoder import WavenetVocoder
from shutil import copy2
from mediaio import ffmpeg
from datetime import datetime
from mediaio.video_io import VideoFileReader
from mediaio.audio_io import AudioSignal
from utils import split_and_concat

BASE_FOLDER = '/cs/labs/peleg/asaph/playground/avse' # todo: remove before releasing code
SPLIT = 6

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
		dataset_path, speaker_ids, args.noise_dirs, max_files=args.number_of_samples
	)

	video_samples, mixed_spectrograms, mixed_phases, source_spectrograms, source_phases = utils.preprocess_data(
		video_file_paths, source_file_paths, noise_file_paths
	)

	np.savez(
		preprocessed_blob_path,
		video_samples=video_samples,
		mixed_spectrograms=mixed_spectrograms,
		mixed_phases=mixed_phases,
		source_spectrograms=source_spectrograms,
		source_phases=source_phases,
	)


def load_preprocessed_samples(preprocessed_blob_paths, max_samples=None):
	all_video_samples = []
	all_mixed_spectrograms = []
	all_source_spectrograms = []
	all_source_phases = []
	all_mixed_phases = []

	for preprocessed_blob_path in preprocessed_blob_paths:
		print('loading preprocessed samples from %s' % preprocessed_blob_path)
		
		with np.load(preprocessed_blob_path) as data:
			all_video_samples.append(data['video_samples'][:max_samples])
			all_mixed_spectrograms.append(data['mixed_spectrograms'][:max_samples])
			all_source_spectrograms.append(data['source_spectrograms'][:max_samples])
			all_source_phases.append(data['source_phases'][:max_samples])
			all_mixed_phases.append(data['mixed_phases'][:max_samples])

	video_samples = np.concatenate(all_video_samples, axis=0)
	mixed_spectrograms = np.concatenate(all_mixed_spectrograms, axis=0)
	source_spectrograms = np.concatenate(all_source_spectrograms, axis=0)
	source_phases = np.concatenate(all_source_phases, axis=0)
	mixed_phases = np.concatenate(all_mixed_phases, axis=0)

	permutation = np.random.permutation(video_samples.shape[0])
	video_samples = video_samples[permutation]
	mixed_spectrograms = mixed_spectrograms[permutation]
	source_spectrograms = source_spectrograms[permutation]
	source_phases = source_phases[permutation]
	mixed_phases = mixed_phases[permutation]

	return (
		video_samples,
		mixed_spectrograms,
		source_spectrograms,
		source_phases,
		mixed_phases
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

	# if args.number_of_samples is None:
	# 	num_train = 900

	train_video_samples, train_mixed_spectrograms, train_source_spectrograms = load_preprocessed_samples(
		train_preprocessed_blob_paths, max_samples=args.number_of_samples
	)[:3]

	val_video_samples, val_mixed_spectrograms, val_source_spectrograms = load_preprocessed_samples(
		val_preprocessed_blob_paths, max_samples=args.number_of_samples
	)[:3]

	print 'normalizing video samples...'
	video_normalizer = utils.VideoNormalizer(train_video_samples)
	video_normalizer.normalize(train_video_samples)
	video_normalizer.normalize(val_video_samples)

	with open(normalization_cache_path, 'wb') as normalization_fd:
		pickle.dump(video_normalizer, normalization_fd)

	num_frames = train_video_samples.shape[3]
	num_audio_bins = num_frames / SPLIT * 4 * SPLIT

	train_mixed_spectrograms = split_and_concat(train_mixed_spectrograms[..., :num_audio_bins], axis=-1, split=SPLIT)
	train_source_spectrograms = split_and_concat(train_source_spectrograms[..., :num_audio_bins], axis=-1, split=SPLIT)
	val_mixed_spectrograms = split_and_concat(val_mixed_spectrograms[..., :num_audio_bins], axis=-1, split=SPLIT)
	val_source_spectrograms = split_and_concat(val_source_spectrograms[..., :num_audio_bins], axis=-1, split=SPLIT)

	train_video_samples = split_and_concat(train_video_samples, axis=-1, split=SPLIT)
	val_video_samples = split_and_concat(val_video_samples, axis=-1, split=SPLIT)

	# transpose freq and time axis
	train_mixed_spectrograms = np.swapaxes(train_mixed_spectrograms, 1, 2)
	train_source_spectrograms = np.swapaxes(train_source_spectrograms, 1, 2)
	val_mixed_spectrograms = np.swapaxes(val_mixed_spectrograms, 1, 2)
	val_source_spectrograms = np.swapaxes(val_source_spectrograms, 1, 2)

	train_video_samples = np.rollaxis(train_video_samples, 3, 1)
	val_video_samples = np.rollaxis(val_video_samples, 3, 1)

	print 'spec shape:', train_mixed_spectrograms.shape
	print 'vid shape:', train_video_samples.shape

	spec_shape = (None, 80)
	video_shape = (None, 128, 128)

	print 'building network...'
	network = SpeechEnhancementNetwork.build(video_shape,
											 spec_shape,
											 num_filters=80,
											 kernel_size=7,
											 num_blocks=20,
											 num_gpus=args.gpus,
											 model_cache_dir=model_cache_dir
											 )
	network.train(
		train_mixed_spectrograms, train_video_samples, train_source_spectrograms,
		val_mixed_spectrograms, val_video_samples, val_source_spectrograms
	)

	# network.save(model_cache_dir)


def predict(args):
	model_cache_dir = os.path.join(args.base_folder, 'cache/models', args.model)
	prediction_output_dir = os.path.join(args.base_folder, 'out', args.model)
	testset_path = os.path.join(args.base_folder, 'cache/preprocessed', args.data_name + '.npz')
	if not os.path.exists(prediction_output_dir):
		os.mkdir(prediction_output_dir)

	vid, mix_specs, source_specs, source_phases, mixed_phases = load_preprocessed_samples([testset_path], max_samples=args.number_of_samples)

	dp = utils.DataProcessor(25, 16000)
	network = SpeechEnhancementNetwork.load(model_cache_dir)


	enhanced_specs = network.predict(np.swapaxes(mix_specs, 1, 2), np.rollaxis(vid, 3, 1))
	enhanced_specs = np.swapaxes(enhanced_specs, 1, 2)

	np.save('/cs/grad/asaph/testing/specs3.npy', enhanced_specs)

	date_dir = os.path.join(prediction_output_dir, datetime.now().strftime('%Y-%m-%d_%H:%M:%S'))
	os.mkdir(date_dir)

	print 'len:', source_specs.shape[2]

	for i in range(enhanced_specs.shape[0]):
		loss = np.sum((enhanced_specs[i] - source_specs[i]) ** 2)

		print i + 1, 'loss:', loss

		enhanced = dp.reconstruct_signal(enhanced_specs[i], mixed_phases[i])
		mixed = dp.reconstruct_signal(mix_specs[i], mixed_phases[i])
		source = dp.reconstruct_signal(source_specs[i], source_phases[i])


		source.save_to_wav_file(os.path.join(date_dir, 'source_' + str(i) + '.wav'))
		mixed.save_to_wav_file(os.path.join(date_dir, 'mixed_' + str(i) + '.wav'))
		enhanced.save_to_wav_file(os.path.join(date_dir, 'enhanced_' + str(i) + '.wav'))

	# model_cache_dir = os.path.join(args.base_folder, 'cache/models', args.model)
	# prediction_output_dir = os.path.join(args.base_folder, 'out', args.model)
	# normalization_cache = os.path.join(model_cache_dir, 'normalization.pkl')
	# dataset_path = os.path.join(args.base_folder, 'data', args.dataset, 'test')
	# if not os.path.exists(prediction_output_dir):
	# 	os.mkdir(prediction_output_dir)
	#
	# storage = PredictionStorage(prediction_output_dir)
	# network = SpeechEnhancementNetwork.load(model_cache_dir)
	#
	# with open(normalization_cache, 'rb') as normalization_fd:
	# 	video_normalizer = pickle.load(normalization_fd)
	#
	# speaker_ids = list_speakers(args)
	# for speaker_id in speaker_ids:
	# 	video_file_paths, speech_file_paths, noise_file_paths = list_data(
	# 		dataset_path, [speaker_id], args.noise_dirs, max_files=5, shuffle=False
	# 	)
	#
	# 	fps = VideoFileReader(video_file_paths[0]).get_frame_rate()
	# 	sr = AudioSignal.from_wav_file(speech_file_paths[0]).get_sample_rate()
	#
	# 	data_processor = utils.DataProcessor(fps, sr)
	#
	# 	for video_file_path, speech_file_path, noise_file_path in zip(video_file_paths, speech_file_paths, noise_file_paths):
	# 		try:
	# 			print('predicting (%s, %s)...' % (video_file_path, noise_file_path))
	# 			mixed_signal = utils.mix_source_noise(speech_file_path, noise_file_path)
	# 			video_samples, mixed_spectrograms, label_spectrograms = data_processor.preprocess_sample(
	# 				video_file_path, speech_file_path, noise_file_path)[:3]
	#
	# 			video_normalizer.normalize(video_samples)
	#
	# 			# loss = network.evaluate(mixed_spectrograms, video_samples, speech_spectrograms)
	# 			# print('loss: %f' % loss)
	# 			enhanced_speech_spectrograms = network.predict(mixed_spectrograms, video_samples)
	#
	# 			enhanced_spec = np.concatenate(list(enhanced_speech_spectrograms), axis=1)
	# 			mixed_spec = data_processor.get_mag_phase(mixed_signal.get_data())[0]
	# 			label_spec = np.concatenate(list(label_spectrograms), axis=1)
	#
	# 			predicted_speech_signal = data_processor.reconstruct_signal(enhanced_spec, mixed_signal, use_griffin_lim=False)
	#
	# 			sample_dir = storage.save_prediction(
	# 				speaker_id, video_file_path, noise_file_path, speech_file_path,
	# 				mixed_signal, predicted_speech_signal, enhanced_spec
	# 			)
	#
	# 			storage.save_spectrograms([enhanced_spec, mixed_spec, label_spec], ['enhanced', 'mixed', 'source'], sample_dir)
	#
	# 		except Exception:
	# 			logging.exception('failed to predict %s. skipping' % video_file_path)


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
		[train_preprocessed_blob_paths], max_samples=20
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
		linear_source_spectrogram = dataProcessor.recover_linear_spectrogram(source_spectrogram)

		waveform = dataProcessor.reconstruct_waveform_data(linear_source_spectrogram, source_phase)

		vocoder_train_enhanced_spectrograms.append(enhanced_spectrogram)
		vocoder_train_enhanced_spectrograms.append(source_spectrogram)
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
	if args.number_of_samples == 0:
		val_preprocessed_blob_path = os.path.join(args.base_folder, 'cache/preprocessed', args.val_data_name + '.npz')

	with np.load(train_preprocessed_blob_path) as data:
		if args.number_of_samples:
			train_enhanced_spectrograms = data['enhanced_spectrograms'][:args.number_of_samples]
			train_waveforms = data['source_waveforms'][:args.number_of_samples]
		else:
			train_enhanced_spectrograms = data['enhanced_spectrograms']
			train_waveforms = data['source_waveforms']

	if args.number_of_samples:
		val_enhanced_spectrograms = train_enhanced_spectrograms
		val_waveforms = train_waveforms
	else:
		with np.load(val_preprocessed_blob_path) as data:
			val_enhanced_spectrograms = data['enhanced_spectrograms']
			val_waveforms = data['source_waveforms']

	train_waveforms = np.c_[train_waveforms, np.zeros((train_waveforms.shape[0], 160))]  # todo: fix net size or label size
	val_waveforms = np.c_[val_waveforms, np.zeros((val_waveforms.shape[0], 160))]

	train_enhanced_spectrograms = np.concatenate(np.split(train_enhanced_spectrograms, SPLIT, axis=2), axis=0)
	train_waveforms = np.concatenate(np.split(train_waveforms, SPLIT, axis=1), axis=0)
	val_enhanced_spectrograms = np.concatenate(np.split(val_enhanced_spectrograms, SPLIT, axis=2), axis=0)
	val_waveforms = np.concatenate(np.split(val_waveforms, SPLIT, axis=1), axis=0)

	print 'building network...'
	network = WavenetVocoder(num_upsample_channels=80,
							 num_dilated_blocks=20,
							 num_skip_channels=128,
							 num_conditioning_channels=10,
							 kernel_size=2,
							 spec_shape=(train_enhanced_spectrograms.shape[1], None),
							 gpus=args.gpus, model_cache_dir=model_cache_dir)
	network.train(train_enhanced_spectrograms, train_waveforms, val_enhanced_spectrograms, val_waveforms)


def predict_vocoder(args):
	model_cache_dir = os.path.join(args.base_folder, 'cache/models', args.model)
	prediction_output_dir = os.path.join(args.base_folder, 'out', args.model)
	testset_path = os.path.join(args.base_folder, 'cache/preprocessed', args.data_name + '.npz')
	if not os.path.exists(prediction_output_dir):
		os.mkdir(prediction_output_dir)

	storage = PredictionStorage(prediction_output_dir)
	network = WavenetVocoder.load(model_cache_dir)

	with np.load(testset_path) as data:
		if args.number_of_samples:
			enhanced_spectrogarms = data['enhanced_spectrograms'][:args.number_of_samples]
			source_waveforms = data['source_waveforms'][:args.number_of_samples]
		else:
			enhanced_spectrogarms = data['enhanced_spectrograms']
			source_waveforms = data['source_waveforms']

	# # todo remove, is here just because I messed us spectrogram shape in train_vocoder (should have been None in time axis)
	# enhanced_spectrogarms = np.concatenate(np.split(enhanced_spectrogarms, 4, axis=2), axis=0)
	# source_waveforms = np.concatenate(np.split(source_waveforms, 4, axis=1), axis=0)

	for i in range(enhanced_spectrogarms.shape[0]):
		print i + 1
		wave_data = network.predict_one_sample(enhanced_spectrogarms[i][np.newaxis, ...])
		enhanced_signal = AudioSignal(wave_data, 16000)
		source_signal = AudioSignal(source_waveforms[i], 16000)

		enhanced_signal.set_sample_type(np.int16)
		source_signal.set_sample_type(np.int16)

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
	preprocess_parser.add_argument('-ns', '--number_of_samples', type=int)
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
	train_parser.add_argument('-ns', '--number_of_samples', type=int)
	train_parser.add_argument('-g', '--gpus', type=int, default=1)
	train_parser.set_defaults(func=train)

	train_vocoder_parser = action_parsers.add_parser('train_vocoder')
	train_vocoder_parser.add_argument('-mn', '--model', type=str, required=True)
	train_vocoder_parser.add_argument('-tdn', '--train_data_name', type=str, required=True)
	train_vocoder_parser.add_argument('-vdn', '--val_data_name', type=str)
	train_vocoder_parser.add_argument('-ns', '--number_of_samples', type=int)
	train_vocoder_parser.add_argument('-g', '--gpus', type=int, default=1)
	train_vocoder_parser.set_defaults(func=train_vocoder)

	predict_parser = action_parsers.add_parser('predict')
	predict_parser.add_argument('-mn', '--model', type=str, required=True)
	predict_parser.add_argument('-dn', '--data_name', type=str, required=True)
	predict_parser.add_argument('-ns', '--number_of_samples', type=int)
	predict_parser.add_argument('-g', '--gpus', type=int, default=1)
	predict_parser.set_defaults(func=predict)

	predict_vocoder_parser = action_parsers.add_parser('predict_vocoder')
	predict_vocoder_parser.add_argument('-mn', '--model', type=str, required=True)
	predict_vocoder_parser.add_argument('-dn', '--data_name', type=str, required=True)
	predict_vocoder_parser.add_argument('-ns', '--number_of_samples', type=int)
	predict_vocoder_parser.add_argument('-g', '--gpus', type=int, default=1)
	predict_vocoder_parser.set_defaults(func=predict_vocoder)

	test_parser = action_parsers.add_parser('test')
	test_parser.add_argument('-mn', '--model', type=str, required=True)
	test_parser.add_argument('-p', '--paths', type=str, nargs='+', required=True)
	test_parser.add_argument('-g', '--gpus', type=int, default=1)
	test_parser.set_defaults(func=test, which='test')

	args = parser.parse_args()
	args.func(args)


if __name__ == '__main__':
	main()
