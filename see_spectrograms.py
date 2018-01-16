
import numpy as np
import matplotlib.pyplot as plt
import librosa as lb
import sys
import os


def plot_spec(path, absolute, show=True):
	spec = np.load(path)
	# if spec.min() >= 0:
	# 	spec = lb.amplitude_to_db(spec)
	if absolute:
		spec = np.abs(spec)

	plt.figure()
	plt.pcolormesh(spec)
	plt.colorbar()
	plt.title(path.split('/')[-1][:-4])
	if show:
		plt.show()

def plot_specs(path, absolute):
	for f in os.listdir(path):
		if f.endswith('npy'):
			file_path = os.path.join(path, f)
			plot_spec(file_path, absolute, show=False)

	plt.show()

def main():
	path = sys.argv[1]
	absolute = sys.argv[2]
	if os.path.isdir(path):
		plot_specs(path, absolute)
	else:
		plot_spec(path, absolute)

if __name__ == "__main__":
	main()