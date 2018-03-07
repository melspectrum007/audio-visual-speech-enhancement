import numpy as np
import librosa as lb
import os, subprocess, multiprocess

from mediaio.audio_io import AudioSignal, AudioMixer
from mediaio.video_io import VideoFileReader
from facedetection.face_detection import FaceDetector

MOUTH_WIDTH = 128
MOUTH_HEIGHT = 128
SAMPLE_RATE = 16000
NUM_MEL_FILTERS = 80
GRIF_LIM_ITERS = 100

class DataProcessor(object):

	def __init__(self, video_fps, audio_sr, db=True, mel=True, audio_bins_per_frame=4):
		self.video_fps = video_fps
		self.audio_sr = audio_sr

		self.mel = mel
		self.db = db

		self.audio_bins_per_frame = audio_bins_per_frame
		self.nfft_single_frame = int(self.audio_sr / self.video_fps)
		self.hop = int(self.nfft_single_frame / self.audio_bins_per_frame)

		self.mean = None
		self.std = None

		if self.mel:
			self.mel_filters = lb.filters.mel(self.audio_sr, self.nfft_single_frame, NUM_MEL_FILTERS, fmin=0, fmax=8000)
			self.invers_mel_filters = np.linalg.pinv(self.mel_filters)

	def preprocess_video(self, frames):
		mouth_cropped_frames = crop_mouth(frames)
		return mouth_cropped_frames

	def get_mag_phase(self, audio_data):
		mag, phase = lb.magphase(lb.stft(audio_data, self.nfft_single_frame, self.hop))

		if self.mel:
			mag = np.dot(self.mel_filters, mag)

		if self.db:
			mag = lb.amplitude_to_db(mag)

		return mag, phase

	# def get_normalization(self, signal):
	# 	self.mean, self.std	= signal.normalize()

	def preprocess_inputs(self, frames, mixed_signal):
		video_sample = self.preprocess_video(frames)
		mixed_spectrogram, mixed_phase = self.get_mag_phase(mixed_signal.get_data())

		return video_sample, mixed_spectrogram, mixed_phase

	def preprocess_label(self, source):
		return self.get_mag_phase(source.get_data())

	def preprocess_sample(self, video_file_path, source_file_path, noise_file_path):
		print ('preprocessing %s, %s' % (source_file_path, noise_file_path))
		frames = get_frames(video_file_path)
		mixed_signal = mix_source_noise(source_file_path, noise_file_path)
		source_signal = AudioSignal.from_wav_file(source_file_path)

		video_sample, mixed_spectrogram, mixed_phase = self.preprocess_inputs(frames, mixed_signal)
		label_spectrogram, label_phase = self.preprocess_label(source_signal)

		return self.truncate_sample_to_same_length(video_sample, mixed_spectrogram, mixed_phase, label_spectrogram, label_phase)

	def truncate_sample_to_same_length(self, video, mixed_spec, mixed_phase, label_spec, label_phase):
		lenghts = [video.shape[-1] * self.audio_bins_per_frame, mixed_spec.shape[-1], mixed_phase.shape[-1], label_spec.shape[-1],
				   label_phase.shape[-1]]

		min_audio_frames = min(lenghts)

		# make sure it divides by audio_bins_per_frame
		min_audio_frames = int(min_audio_frames / self.audio_bins_per_frame) * self.audio_bins_per_frame

		return video[:,:,:min_audio_frames/self.audio_bins_per_frame], mixed_spec[:,:min_audio_frames], mixed_phase[:, :min_audio_frames], \
			   label_spec[:,:min_audio_frames], label_phase[:, :min_audio_frames]


	def try_preprocess_sample(self, sample):
		try:
			return self.preprocess_sample(*sample)
		except Exception as e:
			print('failed to preprocess: %s' % e)
			# traceback.print_exc()
			return None

	def reconstruct_signal(self, spectrogram, phase, use_griffin_lim=False):
		n_frames = min(spectrogram.shape[1], phase.shape[1])
		spectrogram = spectrogram[:, :n_frames]
		phase = phase[:, :n_frames]

		spectrogram = self.recover_linear_spectrogram(spectrogram)

		if use_griffin_lim:
			data = griffin_lim(spectrogram, self.nfft_single_frame, self.hop, GRIF_LIM_ITERS, phase)
		else:
			data = lb.istft(spectrogram * phase, self.hop)

		# data *= self.std
		# data += self.mean
		data = data.astype('int16')
		return AudioSignal(data, self.audio_sr)

	def recover_linear_spectrogram(self, spectrogram):
		if self.db:
			spectrogram = lb.db_to_amplitude(spectrogram)

		if self.mel:
			spectrogram = np.dot(self.invers_mel_filters, spectrogram)

		return spectrogram

	def reconstruct_waveform_data(self, spectrogram, phase):
		return lb.istft(spectrogram * phase, self.hop)


def get_frames(video_path):
	with VideoFileReader(video_path) as reader:
		return reader.read_all_frames(convert_to_gray_scale=True)

def crop_mouth(frames):
	face_detector = FaceDetector()

	mouth_cropped_frames = np.zeros([MOUTH_HEIGHT, MOUTH_WIDTH, frames.shape[0]], dtype=np.float32)
	for i in range(frames.shape[0]):
		mouth_cropped_frames[:, :, i] = face_detector.crop_mouth(frames[i], bounding_box_shape=(MOUTH_WIDTH,
		                                                                                        MOUTH_HEIGHT))
	return mouth_cropped_frames

def mix_source_noise(source_path, noies_path):
	source = AudioSignal.from_wav_file(source_path)
	noise = AudioSignal.from_wav_file(noies_path)

	if source.get_number_of_samples() < noise.get_number_of_samples():
		noise.truncate(source.get_number_of_samples())
	else:
		noise.pad_with_zeros(source.get_number_of_samples())
	noise.amplify(source, 0)

	return AudioMixer().mix([source, noise])

def strip_audio(video_path):
	audio_path = '/tmp/audio.wav'
	subprocess.call(['ffmpeg', '-i', video_path, '-vn', '-acodec', 'copy', audio_path])

	signal = AudioSignal(audio_path, SAMPLE_RATE)
	os.remove(audio_path)

	return signal

def preprocess_data(video_file_paths, source_file_paths, noise_file_paths):
	with VideoFileReader(video_file_paths[0]) as reader:
		fps = reader.get_frame_rate()
	sr = AudioSignal.from_wav_file(source_file_paths[0]).get_sample_rate()
	data_processor = DataProcessor(fps, sr)

	samples = zip(video_file_paths, source_file_paths, noise_file_paths)
	thread_pool = multiprocess.Pool(8)
	preprocessed = thread_pool.map(data_processor.try_preprocess_sample, samples)
	preprocessed = [p for p in preprocessed if p is not None]

	video_samples, mixed_spectrograms, mixed_phases, source_spectrogarms, source_phases = zip(*preprocessed)

	return (
		np.stack(video_samples),
		np.stack(mixed_spectrograms),
		np.stack(mixed_phases),
		np.stack(source_spectrogarms),
		np.stack(source_phases)
	)

def griffin_lim(magnitude, n_fft, hop_length, n_iterations, initial_phase=None):
	"""Iterative algorithm for phase retrival from a magnitude spectrogram."""
	if initial_phase is None:
		phase = np.exp(1j * np.pi * np.random.rand(*magnitude.shape))
	else:
		phase = initial_phase

	signal = lb.istft(magnitude * phase, hop_length=hop_length)

	for i in range(n_iterations):
		D = lb.stft(signal, n_fft=n_fft, hop_length=hop_length)
		phase = lb.magphase(D)[1]

		signal = lb.istft(magnitude * phase, hop_length=hop_length)

	return signal


class VideoNormalizer(object):

	def __init__(self, video_samples):
		# video_samples: slices x height x width x frames_per_slice
		self.__mean_image = np.mean(video_samples, axis=(0, 3))
		self.__std_image = np.std(video_samples, axis=(0, 3))

	def normalize(self, video_samples):
		for s in range(video_samples.shape[0]):
			for f in range(video_samples.shape[3]):
				video_samples[s, :, :, f] -= self.__mean_image
				video_samples[s, :, :, f] /= self.__std_image