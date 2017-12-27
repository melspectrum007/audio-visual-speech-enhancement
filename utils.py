import numpy as np
import librosa as lb
import os, subprocess, multiprocess, traceback, sys

from mediaio.audio_io import AudioSignal, AudioMixer
from mediaio.video_io import VideoFileReader
from facedetection.face_detection import FaceDetector


MOUTH_WIDTH = 128
MOUTH_HEIGHT = 128
BINS_PER_FRAME = 4
SAMPLE_RATE = 16000

class DataProcessor(object):

	def __init__(self, video_fps, audio_sr, num_input_frames=25, num_output_frames=5, mel=False, db=True):
		self.video_fps = video_fps
		self.audio_sr = audio_sr
		self.num_input_frames = num_input_frames
		self.num_output_frames = num_output_frames
		self.input_slice_duration = float(num_input_frames) / self.video_fps
		self.output_slice_duration = float(num_output_frames) / self.video_fps
		self.mel = mel
		self.db = db
		self.nfft_single_frame = int(self.audio_sr / self.video_fps)
		self.hop = int(self.nfft_single_frame / BINS_PER_FRAME)
		self.n_slices = None
		self.mean = None
		self.std = None

	def preprocess_video(self, frames):
		self.n_slices = frames.shape[0] / self.num_output_frames
		frames = frames[:self.n_slices * self.num_output_frames]

		mouth_cropped_frames = crop_mouth(frames)
		pad = (self.num_input_frames - self.num_output_frames) / 2
		mouth_cropped_frames = np.pad(mouth_cropped_frames, ((0, 0), (0, 0), (pad, pad)), 'constant')

		slices = [
			mouth_cropped_frames[:,:, i*self.num_output_frames:i*self.num_output_frames + self.num_input_frames]
			for i in range(self.n_slices)
			]

		return np.stack(slices)

	def slice_input_magphase(self, magphase):
		input_bins_per_slice = self.num_input_frames * BINS_PER_FRAME
		output_bins_per_slice = self.num_output_frames * BINS_PER_FRAME

		pad = (input_bins_per_slice - output_bins_per_slice) / 2
		val = -10 if self.db else 0
		magphase = np.pad(magphase, ((0, 0), (pad, pad), (0, 0)), 'constant', constant_values=val)

		return slice_magphase(magphase, input_bins_per_slice, output_bins_per_slice)

	def get_mag_phase(self, audio_data):

		mag, phase = lb.magphase(lb.stft(audio_data, self.nfft_single_frame, self.hop))

		# if self.mel:
		# 	mel = MelConverter(self.audio_sr, nfft_single_frame, hop, 80, 0, 8000)
		# 	mag = np.dot(mel._MEL_FILTER, mag)

		if self.db:
			mag = lb.amplitude_to_db(mag)

		return np.stack((mag, np.angle(phase)), axis=-1)

	def preprocess_inputs(self, frames, mixed_signal):
		video_samples = self.preprocess_video(frames)

		mixed_magphase = self.get_mag_phase(mixed_signal.get_data())
		mixed_magphases = self.slice_input_magphase(mixed_magphase)

		return video_samples, mixed_magphases

	def preprocess_label(self, source):
		label_magphase = self.get_mag_phase(source.get_data())
		slice_size = self.num_output_frames * BINS_PER_FRAME
		return slice_magphase(label_magphase, slice_size, slice_size)

	def preprocess_sample(self, video_file_path, source_file_path, noise_file_path):
		print ('preprocessing %s, %s' % (source_file_path, noise_file_path))
		frames = get_frames(video_file_path)
		mixed_signal = mix_source_noise(source_file_path, noise_file_path)
		source_signal = AudioSignal.from_wav_file(source_file_path)

		self.mean, self.std = mixed_signal.normalize()
		source_signal.normalize(self.mean, self.std)

		video_samples, mixed_magphases = self.preprocess_inputs(frames, mixed_signal)
		label_magphases = self.preprocess_label(source_signal)

		min_num = min(video_samples.shape[0], mixed_magphases.shape[0])

		return video_samples[:min_num], mixed_magphases[:min_num], label_magphases[:min_num]

	def try_preprocess_sample(self, sample):
		try:
			return self.preprocess_sample(*sample)
		except Exception as e:
			print('failed to preprocess: %s' % e)
			# traceback.print_exc()
			return None

	def reconstruct_signal(self, mag, mixed_signal):

		angle = self.get_mag_phase(mixed_signal.get_data())[:,:,1]
		phase = np.exp(1j * angle)
		phase = phase[:-1, :mag.shape[1]]

		# if self.db:
		# 	mag = lb.db_to_amplitude(mag)
		data = lb.istft(mag * phase, self.hop)
		data *= self.std
		data += self.mean
		data = data.astype('int16')
		return AudioSignal(data, mixed_signal.get_sample_rate())

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

def slice_magphase(magphase, bins_per_slice, hop_length):

	n_slices = (magphase.shape[1] - bins_per_slice) / hop_length + 1

	slices = [
		magphase[:, i * hop_length : i * hop_length + bins_per_slice, :] for i in range(n_slices)
		]

	return np.stack(slices)

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

def preprocess_data(video_file_paths, source_file_paths, noise_file_paths):
	with VideoFileReader(video_file_paths[0]) as reader:
		fps = reader.get_frame_rate()
	sr = AudioSignal.from_wav_file(source_file_paths[0]).get_sample_rate()
	data_processor = DataProcessor(fps, sr)

	samples = zip(video_file_paths, source_file_paths, noise_file_paths)
	thread_pool = multiprocess.Pool(8)
	preprocessed = thread_pool.map(data_processor.try_preprocess_sample, samples)
	preprocessed = [p for p in preprocessed if p is not None]

	video_samples, mixed_magphases, source_magphases = zip(*preprocessed)

	return (
		np.concatenate(video_samples),
		np.concatenate(mixed_magphases),
		np.concatenate(source_magphases)
	)

if __name__ == '__main__':
	dp = DataProcessor(25, 16000)
	s2 = AudioSignal.from_wav_file('/cs/grad/asaph/clean_sounds/lgwm5n_s2.wav')
	m = dp.get_mag_phase(s2.get_data())
	print m.shape