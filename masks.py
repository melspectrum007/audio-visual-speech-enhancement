from mediaio.audio_io import AudioSignal, AudioMixer
import numpy as np
import librosa as lb
import matplotlib.pyplot as plt

s2 = AudioSignal.from_wav_file('/cs/grad/asaph/clean_sounds/lgwm5n_s2.wav')
s3 = AudioSignal.from_wav_file('/cs/grad/asaph/clean_sounds/bgwh6n_s3.wav')
s4 = AudioSignal.from_wav_file('/cs/grad/asaph/clean_sounds/lwiy5p_s4.wav')
s2b = AudioSignal.from_wav_file('/cs/grad/asaph/clean_sounds/pbai6a_s2.wav')
geese = AudioSignal.from_wav_file('/cs/grad/asaph/clean_sounds/geese.wav')

# noises = [s3, s4, geese, s2b]
# clean_spec = lb.magphase(lb.stft(s2.get_data(), 640, 160))[0]
# # log_clean_spec = lb.amplitude_to_db(clean_spec)
# # mask = log_clean_spec > 90
#
# for i, noise in enumerate(noises):
# 	noise.truncate(s2.get_number_of_samples())
# 	noise_spec = lb.magphase(lb.stft(noise.get_data(), 640, 160))[0]
# 	# log_noise_spec = lb.amplitude_to_db(noise_spec)
#
# 	noise.amplify(s2)
# 	mix = AudioMixer.mix([s2, noise]).get_data()
# 	mix_spec, phase = lb.magphase(lb.stft(mix, 640, 160))
# 	# log_mix_spec = lb.amplitude_to_db(mix_spec)
# 	mask = clean_spec > noise_spec
# 	mix_spec *= mask
#
# 	enhanced = lb.istft(mix_spec * phase, 160)
# 	enhanced_sig = AudioSignal(enhanced, 16000)
# 	enhanced_sig.set_sample_type('int16')
# 	enhanced_sig.save_to_wav_file('/cs/grad/asaph/clean_sounds/out' + str(i+1) + '.wav')

# m, p = lb.magphase(lb.stft(s2.get_data(), 640, 160))
# # m = m[:-1, :]
# # p = p[:-1, :]
#
# p = np.roll(p, 300, axis=1)
#
# r = lb.istft(m*p, 160)

data = s2.get_data()
data = data.astype('float64')

# plt.plot(data)
# plt.show()
prelog = lb.amplitude_to_db(lb.magphase(lb.stft(data, 640, 160))[0])

mean = data.mean()
std = data.std()

data -= mean
data /= std
# plt.plot(data)
# plt.show()
postlog = lb.amplitude_to_db(lb.magphase(lb.stft(data, 640, 160))[0])

constlog = lb.amplitude_to_db(lb.magphase(lb.stft(data * 32768, 640, 160))[0])

plt.show()

# a = AudioSignal(r, 16000)
# a.set_sample_type(s2.get_sample_type())
#
# a.save_to_wav_file('/cs/grad/asaph/testing/bbbbb.wav')
