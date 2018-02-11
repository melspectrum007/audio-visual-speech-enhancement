import numpy as np
import librosa as lb
import os, subprocess, multiprocess, traceback, sys

from mediaio.audio_io import AudioSignal, AudioMixer
from mediaio.video_io import VideoFileReader
from facedetection.face_detection import FaceDetector

MOUTH_WIDTH = 128
MOUTH_HEIGHT = 128
SAMPLE_RATE = 16000
NUM_MEL_FILTERS = 80
GRIF_LIM_ITERS = 100

class DataProcessor(object):

	def __init__(self, video_fps, audio_sr, db=True, audio_bins_per_frame=4):
		self.video_fps = video_fps
		self.audio_sr = audio_sr

		self.db = db
		self.audio_bins_per_frame = audio_bins_per_frame
		self.nfft_single_frame = int(self.audio_sr / self.video_fps)
		self.hop = int(self.nfft_single_frame / self.audio_bins_per_frame)

	def get_stft(self, audio_data):

		stft = lb.stft(audio_data, self.nfft_single_frame, self.hop)
		real = stft.real
		imag = stft.imag

		if self.db:
			real = (np.log(np.abs(real) + 1)) * np.sign(real)
			imag = (np.log(np.abs(imag) + 1)) * np.sign(imag)

		return np.stack((real, imag), axis=-1)

	def truncate_sample_to_same_length(self, video_frames, mixed_stft, label_stft):
		video_len = video_frames.shape[-1] * self.audio_bins_per_frame
		mixed_len = mixed_stft.shape[-1]
		label_len = label_stft.shape[-1]

		min_audio_frames = min(video_len, mixed_len, label_len)
		# make sure it divides by audio_bins_per_frame
		min_audio_frames = int(min_audio_frames / self.audio_bins_per_frame) * self.audio_bins_per_frame

		return video_frames[:,:,min_audio_frames/self.audio_bins_per_frame], mixed_stft[:,min_audio_frames,:], label_stft[:,min_audio_frames,:]

	def preprocess_sample(self, video_file_path, source_file_path, noise_file_path):
		print ('preprocessing %s, %s' % (source_file_path, noise_file_path))
		mixed_signal = mix_source_noise(source_file_path, noise_file_path)
		source_signal = AudioSignal.from_wav_file(source_file_path)

		video_frames = get_frames(video_file_path)
		video_frames = np.rollaxis(video_frames, 0, 3) 	# change shape from (num_frames, rows, cols) to (rows, cols, num_frames)
		mixed_stft = self.get_stft(mixed_signal.get_data())
		label_stft = self.get_stft(source_signal.get_data())

		return self.truncate_sample_to_same_length(video_frames, mixed_stft, label_stft)

	def try_preprocess_sample(self, sample):
		try:
			return self.preprocess_sample(*sample)
		except Exception as e:
			print('failed to preprocess: %s' % e)
			traceback.print_exc()
			return None

	def reconstruct_signal(self, stft, sr):
		real = stft[:,:,0]
		imag = stft[:,:,1]

		stft = real + 1j * imag
		data = lb.istft(stft, self.hop)
		data = data.astype('int16')
		return AudioSignal(data, sr)

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
		source.truncate(noise.get_number_of_samples())
	noise.amplify(source, 0)

	return AudioMixer().mix([source, noise])

def strip_audio(video_path):
	audio_path = '/tmp/audio.wav'
	subprocess.call(['ffmpeg', '-i', video_path, '-vn', '-acodec', 'copy', audio_path])

	signal = AudioSignal(audio_path, SAMPLE_RATE)
	os.remove(audio_path)

	return signal

def pad_time_of_short_elements(element_list):
	max_len = max([e.shape[-1] for e in element_list])
	new_list = []
	for e in element_list:
		if e.shape[-1] < max_len:
			new_list.append(np.pad(e, [(0,0)] * (e.ndim - 1) + [(0, max_len - e.shape[-1])], 'constant'))
		else:
			new_list.append(e)

def preprocess_data(video_file_paths, source_file_paths, noise_file_paths):
	with VideoFileReader(video_file_paths[0]) as reader:
		fps = reader.get_frame_rate()
	sr = AudioSignal.from_wav_file(source_file_paths[0]).get_sample_rate()
	data_processor = DataProcessor(fps, sr)

	samples = zip(video_file_paths, source_file_paths, noise_file_paths)
	thread_pool = multiprocess.Pool(8)
	preprocessed = thread_pool.map(data_processor.try_preprocess_sample, samples)
	preprocessed = [p for p in preprocessed if p is not None]

	video_framess, mixed_stfts, source_stfts = zip(*preprocessed)

	return (
		np.stack(pad_time_of_short_elements(video_framess)),
		np.stack(pad_time_of_short_elements(mixed_stfts)),
		np.stack(pad_time_of_short_elements(source_stfts))
	)

