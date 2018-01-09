import numpy as np
import librosa as lb

from mediaio.audio_io import AudioSignal

mixed = AudioSignal.from_wav_file('/cs/grad/asaph/testing/mixture.wav')

mix_spec = np.load('/cs/grad/asaph/testing/mixed.npy')
enhanced_spec = np.load('/cs/grad/asaph/testing/enhanced.npy')

mix_spec = mix_spec[:-1,:280]

mask1 = (enhanced_spec - enhanced_spec.min()) / (enhanced_spec.max() - enhanced_spec.min())

recon1 = mix_spec * lb.db_to_amplitude(enhanced_spec)

recon2 = lb.db_to_amplitude(np.minimum(enhanced_spec, mix_spec))

phase = lb.magphase(lb.stft(mixed.get_data(), 640, 160))[1]
phase = phase[:-1,:280]

s1 = lb.istft(recon1 * phase, 160)
s2 = lb.istft(recon2 * phase, 160)

AudioSignal(s1, 16000).save_to_wav_file('/cs/grad/asaph/testing/recon1.wav')
AudioSignal(s2, 16000).save_to_wav_file('/cs/grad/asaph/testing/recon2.wav')