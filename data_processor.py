import multiprocess
import pickle

import numpy as np

from dsp.spectrogram import MelConverter
from facedetection.face_detection import FaceDetector
from mediaio.audio_io import AudioSignal, AudioMixer
from mediaio.video_io import VideoFileReader


def preprocess_video_sample(video_file_path, slice_duration_ms, mouth_height=128, mouth_width=128):
	print("preprocessing %s" % video_file_path)

	face_detector = FaceDetector()

	with VideoFileReader(video_file_path) as reader:
		frames = reader.read_all_frames(convert_to_gray_scale=True)

		mouth_cropped_frames = np.zeros(shape=(mouth_height, mouth_width, reader.get_frame_count()), dtype=np.float32)
		for i in range(reader.get_frame_count()):
			mouth_cropped_frames[:, :, i] = face_detector.crop_mouth(frames[i], bounding_box_shape=(mouth_width, mouth_height))

		frames_per_slice = int((float(slice_duration_ms) / 1000) * reader.get_frame_rate())
		n_slices = int(float(reader.get_frame_count()) / frames_per_slice)

		slices = [
			mouth_cropped_frames[:, :, (i * frames_per_slice):((i + 1) * frames_per_slice)]
			for i in range(n_slices)
		]

		return np.stack(slices), reader.get_frame_rate()


def preprocess_audio_signal(audio_signal, slice_duration_ms, n_video_slices, video_frame_rate):
	samples_per_slice = int((float(slice_duration_ms) / 1000) * audio_signal.get_sample_rate())
	signal_length = samples_per_slice * n_video_slices

	if audio_signal.get_number_of_samples() < signal_length:
		audio_signal.pad_with_zeros(signal_length)
	else:
		audio_signal.truncate(signal_length)

	n_fft = int(float(audio_signal.get_sample_rate()) / video_frame_rate)
	hop_length = int(n_fft / 4)

	mel_converter = MelConverter(audio_signal.get_sample_rate(), n_fft, hop_length, n_mel_freqs=80, freq_min_hz=0, freq_max_hz=8000)
	mel_spectrogram = mel_converter.signal_to_mel_spectrogram(audio_signal)

	spectrogram_samples_per_slice = int(samples_per_slice / hop_length)
	n_slices = int(mel_spectrogram.shape[1] / spectrogram_samples_per_slice)

	slices = [
		mel_spectrogram[:, (i * spectrogram_samples_per_slice):((i + 1) * spectrogram_samples_per_slice)]
		for i in range(n_slices)
	]

	return np.stack(slices)


def reconstruct_speech_signal(mixed_signal, speech_spectrograms, video_frame_rate):
	n_fft = int(float(mixed_signal.get_sample_rate()) / video_frame_rate)
	hop_length = int(n_fft / 4)

	mel_converter = MelConverter(mixed_signal.get_sample_rate(), n_fft, hop_length, n_mel_freqs=80, freq_min_hz=0, freq_max_hz=8000)
	_, original_phase = mel_converter.signal_to_mel_spectrogram(mixed_signal, get_phase=True)

	speech_spectrogram = np.concatenate(list(speech_spectrograms), axis=1)

	spectrogram_length = min(speech_spectrogram.shape[1], original_phase.shape[1])
	speech_spectrogram = speech_spectrogram[:, :spectrogram_length]
	original_phase = original_phase[:, :spectrogram_length]

	return mel_converter.reconstruct_signal_from_mel_spectrogram(speech_spectrogram, original_phase)


def preprocess_audio_pair(speech_file_path, noise_file_path, slice_duration_ms, n_video_slices, video_frame_rate):
	print("preprocessing pair: %s, %s" % (speech_file_path, noise_file_path))

	speech_signal = AudioSignal.from_wav_file(speech_file_path)
	noise_signal = AudioSignal.from_wav_file(noise_file_path)

	noise_signal.amplify(speech_signal)

	while noise_signal.get_number_of_samples() < speech_signal.get_number_of_samples():
		noise_signal = AudioSignal.concat([noise_signal, noise_signal])

	noise_signal.truncate(speech_signal.get_number_of_samples())

	mixed_signal = AudioMixer.mix([speech_signal, noise_signal], mixing_weights=[1, 0.5])

	speech_spectrograms = preprocess_audio_signal(speech_signal, slice_duration_ms, n_video_slices, video_frame_rate)
	noise_spectrograms = preprocess_audio_signal(noise_signal, slice_duration_ms, n_video_slices, video_frame_rate)
	mixed_spectrograms = preprocess_audio_signal(mixed_signal, slice_duration_ms, n_video_slices, video_frame_rate)

	return mixed_spectrograms, speech_spectrograms, noise_spectrograms, mixed_signal


def preprocess_sample(video_file_path, speech_file_path, noise_file_path, slice_duration_ms=200):
	print("preprocessing sample: %s, %s, %s..." % (video_file_path, speech_file_path, noise_file_path))

	video_samples, video_frame_rate = preprocess_video_sample(video_file_path, slice_duration_ms)
	mixed_spectrograms, speech_spectrograms, noise_spectrograms, mixed_signal = preprocess_audio_pair(
		speech_file_path, noise_file_path, slice_duration_ms, video_samples.shape[0], video_frame_rate
	)

	n_slices = min(video_samples.shape[0], mixed_spectrograms.shape[0])

	return (
		video_samples[:n_slices],
		mixed_spectrograms[:n_slices],
		speech_spectrograms[:n_slices],
		noise_spectrograms[:n_slices],
		mixed_signal,
		video_frame_rate
	)


def try_preprocess_sample(sample):
	try:
		return preprocess_sample(*sample)

	except Exception as e:
		print("failed to preprocess %s (%s)" % (sample, e))
		return None


def preprocess_data(video_file_paths, speech_file_paths, noise_file_paths):
	print("preprocessing data...")

	samples = zip(video_file_paths, speech_file_paths, noise_file_paths)

	thread_pool = multiprocess.Pool(8)
	preprocessed = thread_pool.map(try_preprocess_sample, samples)
	preprocessed = [p for p in preprocessed if p is not None]

	video_samples = [p[0] for p in preprocessed]
	mixed_spectrograms = [p[1] for p in preprocessed]
	speech_spectrograms = [p[2] for p in preprocessed]
	noise_spectrograms = [p[3] for p in preprocessed]

	return (
		np.concatenate(video_samples),
		np.concatenate(mixed_spectrograms),
		np.concatenate(speech_spectrograms),
		np.concatenate(noise_spectrograms)
	)


class DataNormalizer(object):

	@classmethod
	def normalize(cls, video_samples, audio_samples):
		mean_image, std_image = cls.__init_video_normalization_data(video_samples)
		mean_spectrogram, std_spectrogram = cls.__init_audio_normalization_data(audio_samples)

		normalization_data = NormalizationData(mean_image, std_image, mean_spectrogram, std_spectrogram)
		cls.apply_normalization(video_samples, audio_samples, normalization_data)

		return normalization_data

	@classmethod
	def apply_normalization(cls, video_samples, audio_samples, normalization_data):
		cls.apply_video_normalization(video_samples, normalization_data)
		cls.apply_audio_normalization(audio_samples, normalization_data)

	@classmethod
	def apply_video_normalization(cls, video_samples, normalization_data):
		# video_samples: slices x height x width x frames_per_slice
		for s in range(video_samples.shape[0]):
			for f in range(video_samples.shape[3]):
				video_samples[s, :, :, f] -= normalization_data.mean_image
				video_samples[s, :, :, f] /= normalization_data.std_image

	@classmethod
	def apply_audio_normalization(cls, audio_samples, normalization_data):
		for s in range(audio_samples.shape[0]):
			audio_samples[s, :, :] -= normalization_data.mean_spectrogram
			audio_samples[s, :, :] /= normalization_data.std_spectrogram

	@staticmethod
	def __init_video_normalization_data(video_samples):
		# video_samples: slices x height x width x frames_per_slice
		mean_image = np.mean(video_samples, axis=(0, 3))
		std_image = np.std(video_samples, axis=(0, 3))

		return mean_image, std_image

	@staticmethod
	def __init_audio_normalization_data(audio_samples):
		# audio_samples: slices x freqs x time
		mean_spectrogram = np.mean(audio_samples, axis=0)
		std_spectrogram = np.std(audio_samples, axis=0)

		return mean_spectrogram, std_spectrogram


class NormalizationData(object):

	def __init__(self, mean_image, std_image, mean_spectrogram, std_spectrogram):
		self.mean_image = mean_image
		self.std_image = std_image

		self.mean_spectrogram = mean_spectrogram
		self.std_spectrogram = std_spectrogram

	def save(self, path):
		with open(path, 'wb') as normalization_fd:
			pickle.dump(self, normalization_fd)

	@staticmethod
	def load(path):
		with open(path, 'rb') as normalization_fd:
			return pickle.load(normalization_fd)
