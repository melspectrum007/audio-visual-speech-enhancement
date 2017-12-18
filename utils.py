import numpy as np
import librosa as lb
import os
import subprocess

from mediaio.audio_io import AudioSignal, AudioMixer
from mediaio.video_io import VideoFileReader
from facedetection.face_detection import FaceDetector
from dsp.spectrogram import MelConverter

MOUTH_WIDTH = 128
MOUTH_HEIGHT = 128
BINS_PER_FRAME = 4
SAMPLE_RATE = 16000

class DataProcessor(object):

	def __init__(self, example_video_path, example_audio_path, num_input_frames=11, num_output_frames=5, mel=False, db=True):
		with VideoFileReader(example_video_path) as reader:
			self.video_fps = reader.get_frame_rate()
			self.frame_count = reader.get_frame_count()

		self.audio_sr = AudioSignal.from_wav_file(example_audio_path).get_sample_rate()
		self.num_input_frames = num_input_frames
		self.num_output_frames = num_output_frames
		self.input_slice_duration = float(num_input_frames) / self.video_fps
		self.output_slice_duration = float(num_output_frames) / self.video_fps
		self.mel = mel
		self.db = db
		self.nfft_single_frame = self.audio_sr / self.video_fps
		self.hop = int(self.nfft_single_frame / BINS_PER_FRAME)
		self.n_slices = int(float(self.frame_count) / self.num_input_frames)

	def slice_video(self, video_path):
		print('slicing %s' % video_path)

		face_detector = FaceDetector()
		with VideoFileReader(video_path) as reader:
			frames = reader.read_all_frames(convert_to_gray_scale=True)

			mouth_cropped_frames = np.zeros([MOUTH_HEIGHT, MOUTH_WIDTH, reader.get_frame_count()], dtype=np.float32)
			for i in range(reader.get_frame_count()):
				mouth_cropped_frames[:, :, i] = face_detector.crop_mouth(frames[i], bounding_box_shape=(MOUTH_WIDTH, MOUTH_HEIGHT))

			pad = (self.num_input_frames - self.num_output_frames) / 2
			mouth_cropped_frames = np.pad(mouth_cropped_frames, ((0, 0), (0, 0), (pad, pad)), 'constant')

			slices = [
				mouth_cropped_frames[:, :, (i * self.num_input_frames):((i + 1) * self.num_input_frames)]
				for i in range(self.n_slices)
			]

		return np.stack(slices)

	def get_mag_phase(self, data):

		mag, phase = lb.magphase(lb.stft(data, self.nfft_single_frame, self.hop))

		# if self.mel:
		# 	mel = MelConverter(self.audio_sr, nfft_single_frame, hop, 80, 0, 8000)
		# 	mag = np.dot(mel._MEL_FILTER, mag)

		if self.db:
			mag = lb.amplitude_to_db(mag)

		return mag, phase

	def slice_input_spectrogram(self, spectrogram):
		input_bins_per_slice = self.num_input_frames * BINS_PER_FRAME
		output_bins_per_slice = self.num_output_frames * BINS_PER_FRAME

		pad = (input_bins_per_slice - output_bins_per_slice) / 2
		val = -10 if self.db else 0
		spectrogram = np.pad(spectrogram, ((0, 0), (pad, pad)), 'constant', constant_values=val)

		return DataProcessor.slice_spectrogram(spectrogram, self.n_slices, input_bins_per_slice)

	@staticmethod
	def slice_spectrogram(spectrogram, n_slices, bins_per_slice):

		slices = [
			spectrogram[:, i * bins_per_slice : (i+1) * bins_per_slice] for i in range(n_slices)
		]

		return np.stack(slices)

	@staticmethod
	def mix_source_noise(source_path, noies_path):
		source = AudioSignal.from_wav_file(source_path)
		noise = AudioSignal.from_wav_file(noies_path)

		noise.truncate(source.get_number_of_samples())
		noise.amplify(source)

		return AudioMixer([source, noise])

	@staticmethod
	def strip_audio(video_path):
		audio_path = '/tmp/audio.wav'
		subprocess.call(['ffmpeg', '-i', video_path, '-vn', '-acodec', 'copy', audio_path])

		signal = AudioSignal(audio_path, SAMPLE_RATE)
		os.remove(audio_path)

		return signal

	def preprocess_inputs(self, video_path, mixed_audio_path):
		video_samples = self.slice_video(video_path)

		mixed_signal = AudioSignal.from_wav_file(mixed_audio_path)
		mixed_spectrogram = self.get_mag_phase(mixed_signal.get_data())
		mixed_spectrograms = self.slice_input_spectrogram(mixed_spectrogram)

		return video_samples, mixed_spectrograms

	def preprocess_label(self, source_path):
		label_spectrogram = self.get_mag_phase(AudioSignal.from_wav_file(source_path).get_data())[0]

		return DataProcessor.slice_spectrogram(label_spectrogram, self.n_slices, self.num_output_frames * BINS_PER_FRAME)


