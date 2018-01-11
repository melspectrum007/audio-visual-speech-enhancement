import numpy as np
from utils import DataProcessor

from mediaio.audio_io import AudioSignal

dp = DataProcessor(25, 16000)

mixed = AudioSignal.from_wav_file('/cs/grad/asaph/testing/mixture.wav')
source = AudioSignal.from_wav_file('/cs/grad/asaph/testing/source.wav')

mix_spec = np.load('/cs/grad/asaph/testing/mixed.npy')
enhanced_spec = np.load('/cs/grad/asaph/testing/enhanced.npy')
source_spec = np.load('/cs/grad/asaph/testing/source.npy')

dp.get_normalization(mixed)
enhanced_spec_mixed_phase = dp.reconstruct_signal(enhanced_spec, mixed)
enhanced_spec_source_phase = dp.reconstruct_signal(enhanced_spec, source)
source_spec_mixed_phase = dp.reconstruct_signal(source_spec, mixed)

enhanced_spec_mixed_phase.save_to_wav_file('/cs/grad/asaph/testing/enhanced_spec_mixed_phase')
enhanced_spec_source_phase.save_to_wav_file('/cs/grad/asaph/testing/enhanced_spec_source_phase')
source_spec_mixed_phase.save_to_wav_file('/cs/grad/asaph/testing/source_spec_mixed_phase')



