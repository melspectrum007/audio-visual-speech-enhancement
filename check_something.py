
from mediaio.audio_io import AudioSignal, AudioMixer
import numpy as np

from speech_enhancer import load_preprocessed_samples
from utils import DataProcessor

dp = DataProcessor(25, 16000)

ds = np.load('/cs/labs/peleg/asaph/playground/avse/cache/preprocessed/s2-log-mel-unnorm-train-vocoder.npz')

enhanced_specs = ds['enhanced_spectrograms'][:20]

ds = np.load('/cs/labs/peleg/asaph/playground/avse/cache/preprocessed/s2-log-mel-unnorm-train.npz')

source_phases = ds['source_phases'][:20]

for i in range(20):
	spec = dp.recover_linear_spectrogram(enhanced_specs[i])
	waveform = dp.reconstruct_waveform_data(spec, source_phases[i])
	s = AudioSignal(waveform, 16000)
	s.set_sample_type(np.int16)
	s.save_to_wav_file('/cs/grad/asaph/testing/enhnced_spec_source_phase_' + str(i) + '.wav')


