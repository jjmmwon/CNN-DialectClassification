import spectrogram as sp


label_file_path = './gsd_label'
wav_file_path = './gsd_wav'
save_regind_dir = './gsd_preprocessed'

spec_preprocessing = sp.Spectrogram(label_file_path='../gsd_label', wav_file_path='../gsd_wav', save_region_dir='../gsd_preprocessed', save_image_dir='../gsd_vis', vis=10)

# hop_length is length overlapped between adjacent window, It is usally set to (sample_rate/100)
# n_fft is length of windowed signal after padding zeros, It is usally set to (sample_rate/40)

hop_length = 160
n_fft = 400
sample_rate = 16000

spec_preprocessing.spectrogram(hop_length=hop_length, n_fft=n_fft, sr=sample_rate)
