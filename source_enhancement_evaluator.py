import argparse
import os
import subprocess
import tempfile
import re
from shutil import copy2
import numpy as np

from mediaio import ffmpeg


def pesq(pesq_bin_path, source_file_path, enhanced_file_path):
	temp_dir = tempfile.gettempdir()

	temp_source_path = os.path.join(temp_dir, 'source.wav')
	temp_enhanced_path = os.path.join(temp_dir, 'enhanced.wav')

	copy2(source_file_path, temp_source_path)
	copy2(enhanced_file_path, temp_enhanced_path)

	# ffmpeg.downsample(source_file_path, temp_source_path, sample_rate=16000)
	# ffmpeg.downsample(enhanced_file_path, temp_estimated_path, sample_rate=16000)

	try:
		output = subprocess.check_output(
			[pesq_bin_path, "+16000", temp_source_path, temp_enhanced_path]
		)
	except Exception:
		print 'exception'
		return None, None

	# print output

	match = re.search("\(Raw MOS, MOS-LQO\):\s+= (-?[0-9.]+?)\s+([0-9.]+?)$", output, re.MULTILINE)
	mos = float(match.group(1))
	moslqo = float(match.group(2))

	# os.remove(temp_source_path)
	# os.remove(temp_estimated_path)

	return mos, moslqo


def evaluate(enhancement_dir_path, pesq_bin_path):
	enhanced_pesqs = []
	mixture_pesqs = []

	speaker_dir_names = os.listdir(enhancement_dir_path)
	for speaker_dir_name in speaker_dir_names:
		speaker_dir_path = os.path.join(enhancement_dir_path, speaker_dir_name)
		sample_dir_names = sorted(os.listdir(speaker_dir_path))
		for sample_dir_name in sample_dir_names:
			print sample_dir_name
			sample_dir_path = os.path.join(speaker_dir_path, sample_dir_name)
			source_file_path = os.path.join(sample_dir_path, "source.wav")
			enhanced_file_path = os.path.join(sample_dir_path, "enhanced.wav")
			mixture_file_path = os.path.join(sample_dir_path, "mixture.wav")

			enhanced_mos, _ = pesq(pesq_bin_path, source_file_path, enhanced_file_path)
			mixture_mos, _ = pesq(pesq_bin_path, source_file_path, mixture_file_path)

			print 'mixture pesq: ', mixture_mos, 'enhanced pesq: ', enhanced_mos
			if enhanced_mos is None or mixture_mos is None:
				continue
			else:
				enhanced_pesqs.append(enhanced_mos)
				mixture_pesqs.append(mixture_mos)

	print 'mean enhanced pesq: ', np.mean(enhanced_pesqs), 'std enhanced pesq: ', np.std(enhanced_pesqs)
	print 'mean mixture pesq: ', np.mean(mixture_pesqs), 'std mixture pesq: ', np.std(mixture_pesqs)
	# print('pesqs', enhanced_pesqs)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("enhancement_dir", type=str)
	parser.add_argument("pesq_bin_path", type=str)
	args = parser.parse_args()

	evaluate(args.enhancement_dir, args.pesq_bin_path)


if __name__ == "__main__":
	main()
