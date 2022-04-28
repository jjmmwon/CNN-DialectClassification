import spectrogram as sp


label_file_path = './gsd_label'
wav_file_path = './gsd_wav'
save_regind_dir = './gsd_preprocessed'

spec_preprocessing = sp.Spectrogram(label_file_path='./gsd_label', wav_file_path='./gsd_wav', save_region_dir='./gsd_preprocessed')

hop_length = 160
n_fft = 400
sample_rate = 16000

spec_preprocessing.spectrogram(hop_length=hop_length, n_fft=n_fft, sr=sample_rate)
