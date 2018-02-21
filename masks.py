from mediaio.audio_io import AudioSignal, AudioMixer
import numpy as np
import librosa as lb
import matplotlib.pyplot as plt

s = AudioSignal.from_wav_file('/cs/grad/asaph/testing/source_0.wav')
e = AudioSignal.from_wav_file('/cs/grad/asaph/testing/enhanced_0.wav')

ss = s.get_data()
ee = e.get_data()

# ee[ee==-32768] = 0

# AudioSignal(ee, 16000).save_to_wav_file('/cs/grad/asaph/testing/recon_0.wav')

plt.plot(ss)
plt.plot(ee)

plt.show()
