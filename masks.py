from mediaio.audio_io import AudioSignal, AudioMixer
import numpy as np
import librosa as lb
import matplotlib.pyplot as plt

# s2 = AudioSignal.from_wav_file('/cs/grad/asaph/clean_sounds/lgwm5n_s2.wav')
s2 = AudioSignal.from_wav_file('/cs/grad/asaph/examples/mask-predict/source.wav')
enh = AudioSignal.from_wav_file('/cs/grad/asaph/examples/mask-predict/enhanced.wav')
mixture = AudioSignal.from_wav_file('/cs/grad/asaph/examples/mask-predict/mixture.wav')
# s3 = AudioSignal.from_wav_file('/cs/grad/asaph/clean_sounds/bgwh6n_s3.wav')
# s4 = AudioSignal.from_wav_file('/cs/grad/asaph/clean_sounds/lwiy5p_s4.wav')
# s2b = AudioSignal.from_wav_file('/cs/grad/asaph/clean_sounds/pbai6a_s2.wav')
# geese = AudioSignal.from_wav_file('/cs/grad/asaph/clean_sounds/geese.wav')

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

# st = lb.stft(mixture.get_data(), 640, 160)
# re = st.real
# im = st.imag
#
# plt.figure()
# plt.pcolormesh(lb.amplitude_to_db(re))
# plt.figure()
# plt.pcolormesh(lb.amplitude_to_db(im))
# plt.figure()
# plt.pcolormesh(lb.amplitude_to_db(lb.magphase(st)[0]))
# plt.show()


mix_st = lb.stft(mixture.get_data(), 640, 160)

mix_st = mix_st[:,:280]
m1, p1 = lb.magphase(lb.stft(s2.get_data(), 640, 160))
m2, p2 = lb.magphase(lb.stft(enh.get_data(), 640, 160))

# re = mix_st.real
# im = mix_st.imag
# m = m[:-1, :]
# p = p[:-1, :]

mag_s, phase_s = lb.magphase(lb.stft(s2.get_data(), 640, 160))
mag_e, phase_e = lb.magphase(lb.stft(enh.get_data(), 640, 160))
mag_m, phase_m = lb.magphase(mix_st)

# mask = mag_e / mag_m

# mag_e = mag_e[:-1, :]
# mag_m = mag_m[:-1, :]

# stft = re * mask + im * mask * 1j

# r = lb.istft(stft, 160)
# m1 = m1[:, :p2.shape[1]]

r = lb.istft(mag_e * phase_s[:,:mag_e.shape[1]], 160)
a = AudioSignal(r, 16000)
a.set_sample_type(s2.get_sample_type())
a.save_to_wav_file('/cs/grad/asaph/examples/mask-predict/enhanced-mag-source-phase.wav')

r = lb.istft(mag_m * phase_s[:,:mag_m.shape[1]], 160)
a = AudioSignal(r, 16000)
a.set_sample_type(s2.get_sample_type())
a.save_to_wav_file('/cs/grad/asaph/examples/mask-predict/mixed-mag-source-phase.wav')

r = lb.istft(mag_s[:,:phase_m.shape[1]] * phase_m, 160)
a = AudioSignal(r, 16000)
a.set_sample_type(s2.get_sample_type())
a.save_to_wav_file('/cs/grad/asaph/examples/mask-predict/source-mag-mixed-phase.wav')

def invert_magnitude_phase(magnitude, phase_angle):
	phase = np.cos(phase_angle) + 1.j * np.sin(phase_angle)
	return magnitude * phase


def griffin_lim(magnitude, n_fft, hop_length, n_iterations):
	"""Iterative algorithm for phase retrival from a magnitude spectrogram."""
	phase_angle = np.pi * np.random.rand(*magnitude.shape)
	D = invert_magnitude_phase(magnitude, phase_angle)
	signal = lb.istft(D, hop_length=hop_length)

	for i in range(n_iterations):
		D = lb.stft(signal, n_fft=n_fft, hop_length=hop_length)
		_, phase = lb.magphase(D)
		phase_angle = np.angle(phase)

		D = invert_magnitude_phase(magnitude, phase_angle)
		signal = lb.istft(D, hop_length=hop_length)

	return signal

r = griffin_lim(mag_e, 640, 160, 50)
a = AudioSignal(r, 16000)
a.set_sample_type(s2.get_sample_type())


a.save_to_wav_file('/cs/grad/asaph/examples/mask-predict/enhanced-mag-griffin_lim-phase.wav')
