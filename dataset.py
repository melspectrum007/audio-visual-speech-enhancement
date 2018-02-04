import os
import glob
import random


class AudioVisualDataset:

	def __init__(self, base_path):
		self._base_path = base_path

	def subset(self, speaker_ids, max_files=None, shuffle=False):
		audio_file_paths = []

		for speaker_id in speaker_ids:
			audio_file_paths.extend(glob.glob(os.path.join(self._base_path, speaker_id, "audio", "*.wav")))

		if shuffle:
			random.shuffle(audio_file_paths)

		return AudioVisualSubset(audio_file_paths[:max_files])

	def list_speakers(self):
		return os.listdir(self._base_path)


class AudioVisualSubset:

	def __init__(self, audio_file_paths):
		self._audio_file_paths = audio_file_paths

	def audio_paths(self):
		return self._audio_file_paths

	def video_paths(self):
		return [self._audio_to_video_path(a) for a in self._audio_file_paths]

	def size(self):
		return len(self._audio_file_paths)

	@staticmethod
	def _audio_to_video_path(audio_file_path):
		return glob.glob(os.path.splitext(audio_file_path.replace('/audio/', '/video/'))[0] + ".*")[0]


class AudioDataset:

	def __init__(self, base_paths):
		self._base_paths = base_paths

	def subset(self, max_files=None, shuffle=False):
		audio_file_paths = [os.path.join(d, f) for d in self._base_paths for f in os.listdir(d)]

		if shuffle:
			random.shuffle(audio_file_paths)

		return audio_file_paths[:max_files]
