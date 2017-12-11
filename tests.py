from mediaio.audio_io import AudioSignal, AudioMixer
from dsp.spectrogram import MelConverter
import librosa

signal = AudioSignal.from_wav_file('/cs/labs/peleg/asaph/playground/data/grid/train/s2/audio/lwiy8a.wav')
noise = AudioSignal.from_wav_file('/cs/labs/peleg/asaph/playground/data/noise/City_Night_Crowd.wav')

noise.truncate(signal.get_number_of_samples())

noise.amplify(signal)
mix = AudioMixer.mix([signal, noise])

mel = MelConverter(16000, 640, 160, 80, 0, 8000)

sig_spec = librosa.core.magphase(librosa.stft(signal.get_data(), 640, 160))[0]
noise_spec = librosa.core.magphase(librosa.stft(noise.get_data(), 640, 160))[0]
mix_spec, phase = librosa.core.magphase(librosa.stft(mix.get_data(), 640, 160))

sig_mel_spec, p = mel.signal_to_mel_spectrogram(signal, get_phase=True)
noise_mel_spec = mel.signal_to_mel_spectrogram(noise)
mix_mel_spec = mel.signal_to_mel_spectrogram(mix)

irm = sig_spec**2 / (sig_spec**2 + noise_spec**2)
smm = sig_spec / mix_spec
log_smm = mel.sectogram_to_mel(smm)


irm_spec = irm * mix_spec
smm_spec = smm * mix_spec
mix_mel_spec += log_smm

irm_clean = mel.reconstruct_signal_from_spectrogram(irm_spec, phase, mel=False)
smm_clean = mel.reconstruct_signal_from_spectrogram(smm_spec, phase, mel=False)
log_smm_clean = mel.reconstruct_signal_from_spectrogram(mix_mel_spec, phase)

irm_clean.save_to_wav_file('/cs/grad/asaph/testing/irm_clean.wav')
smm_clean.save_to_wav_file('/cs/grad/asaph/testing/smm_clean.wav')
log_smm_clean.save_to_wav_file('/cs/grad/asaph/testing/log_smm_clean.wav')
mix.save_to_wav_file('/cs/grad/asaph/testing/mix.wav')