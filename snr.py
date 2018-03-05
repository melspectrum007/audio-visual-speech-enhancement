
import numpy as np
from mediaio.audio_io import AudioSignal, AudioMixer

def rms(x):
	return np.sqrt(np.mean(x**2))

signal = AudioSignal.from_wav_file('/cs/labs/peleg/asaph/playground/data/grid/test/s2/audio/bbaf2s.wav')
noise = AudioSignal.from_wav_file('/cs/labs/peleg/asaph/playground/data/grid/test/s3b/audio/pbao7s.wav')

# s = signal.get_data()
# n = noise.get_data()
#
# eq = rms(s) / rms(n)
#
# data = (eq * n + s).astype(signal.get_sample_type())
# AudioSignal(data, signal.get_sample_rate()).save_to_wav_file('/cs/grad/asaph/testing/equal.wav')

snrs_db = np.linspace(-10, 10, 5)

for snr_db in snrs_db:
	mixed = AudioMixer.snr_mix(signal, noise, snr_db)

	# factor = eq * (10 ** (-snr_db / 20))
	# data = (s + n * factor).astype(signal.get_sample_type())
	mixed.save_to_wav_file('/cs/grad/asaph/testing/snr_db_' + str(snr_db) + '.wav')

