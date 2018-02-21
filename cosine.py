import numpy as np
from utils import DataProcessor

from mediaio.audio_io import AudioSignal
import matplotlib.pyplot as plt

def mu_law_quantization(x, mu, max_val=None):
	x = x.astype('f')
	if max_val is None:
		max_val = np.abs(x).max()

	x = np.clip(x, -max_val, max_val - 1)
	x /= max_val

	y = np.sign(x) * np.log(1 + np.abs(x) * (mu - 1)) / np.log(1 + (mu - 1))
	bins = np.linspace(-1, 1, mu + 1)
	return np.digitize(y, bins) - 1


def one_hot_encoding(y, num_classes):
	one_hot = np.zeros((y.shape[0], num_classes, y.shape[1]))
	one_hot[np.arange(one_hot.shape[0])[:, np.newaxis], y, np.arange(one_hot.shape[2])] = 1

	return one_hot
# mixed = AudioSignal.from_wav_file('/cs/grad/asaph/testing/mixture.wav')
source = AudioSignal.from_wav_file('/cs/grad/asaph/testing/source_0.wav')

s = source.get_data()
q = mu_law_quantization(s, 256, 32768)

h = one_hot_encoding(q[np.newaxis,:], 256)
bins = np.linspace(-1, 1, 256 + 1)[:-1]
classes = np.argmax(h, axis=1)
y = bins[classes]
norm_waveform = np.sign(y) * ((1 + 255) ** np.abs(y) - 1) / 255
r = np.squeeze(norm_waveform) * 32768

# recon = AudioSignal(r, 16000)
# recon.set_sample_type(np.int16)
# recon.save_to_wav_file('/cs/grad/asaph/testing/recon.wav')

# plt.plot(s)
# plt.figure()
plt.plot(q)
plt.figure()
plt.pcolormesh(np.squeeze(h))

plt.show()

print 'bye!'




