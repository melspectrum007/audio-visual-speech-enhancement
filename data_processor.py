import math
import multiprocess

import numpy as np

from dsp.spectrogram import MelConverter
from facedetection.face_detection import FaceDetector
from mediaio.audio_io import AudioSignal, AudioMixer
from mediaio.video_io import VideoFileReader


def preprocess_video_sample(video_file_path, slice_duration_ms=330, mouth_height=50, mouth_width=100):
	print("preprocessing %s" % video_file_path)

	face_detector = FaceDetector()

	with VideoFileReader(video_file_path) as reader:
		frames = reader.read_all_frames()

		mouth_cropped_frames = np.zeros(shape=(reader.get_frame_count(), mouth_height, mouth_width, 3), dtype=np.float32)
		for i in range(reader.get_frame_count()):
			mouth_cropped_frames[i, :] = face_detector.crop_mouth(frames[i, :], bounding_box_shape=(mouth_width, mouth_height))

		frames_per_slice = (float(slice_duration_ms) / 1000) * reader.get_frame_rate()
		n_slices = int(float(reader.get_frame_count()) / frames_per_slice)

		slices = [
			mouth_cropped_frames[int(i * frames_per_slice) : int(math.ceil((i + 1) * frames_per_slice))]
			for i in range(n_slices)
		]

		return np.stack(slices)


def try_preprocess_video_sample(video_file_path):
	try:
		return preprocess_video_sample(video_file_path)

	except Exception as e:
		print("failed to preprocess %s (%s)" % (video_file_path, e))
		return None


def preprocess_audio_signal(audio_signal, slice_duration_ms=330):
	new_signal_length = int(math.ceil(
		float(audio_signal.get_number_of_samples()) / MelConverter.HOP_LENGTH
	)) * MelConverter.HOP_LENGTH

	audio_signal.pad_with_zeros(new_signal_length)

	mel_converter = MelConverter(audio_signal.get_sample_rate(), n_mel_freqs=128, freq_min_hz=0, freq_max_hz=4000)
	mel_spectrogram = mel_converter.signal_to_mel_spectrogram(audio_signal)

	samples_per_slice = int((float(slice_duration_ms) / 1000) * audio_signal.get_sample_rate())
	spectrogram_samples_per_slice = int(samples_per_slice / MelConverter.HOP_LENGTH)

	n_slices = int(mel_spectrogram.shape[1] / spectrogram_samples_per_slice)

	slices = [
		mel_spectrogram[:, (i * spectrogram_samples_per_slice):((i + 1) * spectrogram_samples_per_slice)]
		for i in range(n_slices)
	]

	return np.stack(slices)


def preprocess_audio_pair(speech_file_path, noise_file_path):
	print("preprocessing pair: %s, %s" % (speech_file_path, noise_file_path))

	speech_signal = AudioSignal.from_wav_file(speech_file_path)
	noise_signal = AudioSignal.from_wav_file(noise_file_path)
	mixed_signal = AudioMixer.mix([speech_signal, noise_signal])

	speech_spectrograms = preprocess_audio_signal(speech_signal)
	noise_spectrograms = preprocess_audio_signal(noise_signal)
	mixed_spectrograms = preprocess_audio_signal(mixed_signal)

	speech_masks = np.zeros(shape=mixed_spectrograms.shape)
	speech_masks[speech_spectrograms > noise_spectrograms] = 1

	return mixed_spectrograms, speech_masks, mixed_signal


def reconstruct_speech_signal(mixed_spectrograms, speech_masks, sample_rate):
	mixed_spectrograms = [mixed_spectrograms[i] for i in range(mixed_spectrograms.shape[0])]
	full_mixed_spectrogram = np.concatenate(mixed_spectrograms, axis=1)

	speech_masks = [speech_masks[i] for i in range(speech_masks.shape[0])]
	full_speech_mask = np.concatenate(speech_masks, axis=1)

	full_speech_mask[full_speech_mask >= 0.5] = 1
	full_speech_mask[full_speech_mask < 0.5] = 0

	speech_spectrogram = full_mixed_spectrogram * full_speech_mask

	mel_converter = MelConverter(sample_rate, n_mel_freqs=128, freq_min_hz=0, freq_max_hz=4000)
	return mel_converter.reconstruct_signal_from_mel_spectrogram(speech_spectrogram)


def preprocess_audio_data(speech_file_paths, noise_file_paths):
	print("preprocessing audio data...")

	audio_pairs = zip(speech_file_paths, noise_file_paths)

	thread_pool = multiprocess.Pool(8)
	preprocessed_pairs = thread_pool.map(lambda pair: preprocess_audio_pair(pair[0], pair[1]), audio_pairs)

	mixed_samples = [p[0] for p in preprocessed_pairs]
	speech_mask_samples = [p[1] for p in preprocessed_pairs]

	return np.concatenate(mixed_samples), np.concatenate(speech_mask_samples)


def preprocess_video_data(video_file_paths):
	print("preprocessing video data...")

	thread_pool = multiprocess.Pool(8)
	video_samples = thread_pool.map(try_preprocess_video_sample, video_file_paths)

	invalid_sample_ids = [i for i, sample in enumerate(video_samples) if sample is None]
	video_samples = [sample for i, sample in enumerate(video_samples) if i not in invalid_sample_ids]

	return np.concatenate(video_samples)


class Normalizer:

	@classmethod
	def normalize(cls, audio_samples, video_samples):
		audio_normalization_data = cls.__init_audio_normalization_data(audio_samples)
		video_normalization_data = cls.__init_video_normalization_data(video_samples)

		normalization_data = NormalizationData(audio_normalization_data, video_normalization_data)
		cls.apply_normalization(audio_samples, video_samples, normalization_data)

		return normalization_data

	@classmethod
	def apply_normalization(cls, audio_samples, video_samples, normalization_data):
		cls.__normalize_audio_samples(audio_samples, normalization_data.audio_normalization_data)
		cls.__normalize_video_samples(video_samples, normalization_data.video_normalization_data)

	@staticmethod
	def __init_audio_normalization_data(audio_samples):
		return AudioNormalizationData(
			np.mean(audio_samples),
			np.std(audio_samples - np.mean(audio_samples))
		)

	@staticmethod
	def __init_video_normalization_data(video_samples):
		# video_samples: slices x frames_per_slice x height x width x channels
		channel_means = [video_samples[:, :, :, :, channel].mean() / 255 for channel in range(3)]

		return VideoNormalizationData(channel_means)

	@staticmethod
	def __normalize_audio_samples(audio_samples, audio_normalization_data):
		audio_samples -= audio_normalization_data.mean
		audio_samples /= audio_normalization_data.std

	@staticmethod
	def __normalize_video_samples(video_samples, video_normalization_data):
		video_samples /= 255

		for channel in range(3):
			video_samples[:, :, :, :, channel] -= video_normalization_data.channel_means[channel]


class NormalizationData:

	def __init__(self, audio_normalization_data, video_normalization_data):
		self.audio_normalization_data = audio_normalization_data
		self.video_normalization_data = video_normalization_data


class AudioNormalizationData:

	def __init__(self, mean, std):
		self.mean = mean
		self.std = std


class VideoNormalizationData:

	def __init__(self, channel_means):
		self.channel_means = channel_means
