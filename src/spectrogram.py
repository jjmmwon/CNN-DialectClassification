import numpy as np
import librosa, librosa.display
import json
import pickle
import matplotlib.pyplot as plt

import os

class Spectrogram:


    # label_file_path : the path of the directory include label json files
    # wav_file_path : the path of the directory include wav files
    # save_region_dir : the path of the directory to save preprocessed data as pickle
    # save_image_dir : the path of the directory to save spectrogram plot
    # vis : the number of image files want to plot

    def __init__(self, label_file_path, wav_file_path, save_region_dir, save_image_dir,vis):
        self.label_file_path = label_file_path
        self.wav_file_path = wav_file_path
        self.save_region_dir = save_region_dir
        self.save_image_dir = save_image_dir
        self.vis = vis

        json_file_names = os.listdir(label_file_path)
        json_file_names = [j for j in json_file_names if j.endswith('.json')]

        self.json_file_names = json_file_names

        wav_file_names = os.listdir(wav_file_path)
        wav_file_names = [w for w in wav_file_names if w.endswith('.wav')]

        self.wav_file_names = wav_file_names

        wav_label_dict = dict()

        for json_file_name in json_file_names:
            matched_wav_file = list(filter(lambda x: x[:-3] == json_file_name[:-4], wav_file_names))

            if len(matched_wav_file) == 0: continue
            
            wav_label_dict[json_file_name] = matched_wav_file[0]
        
        self.wav_label_dict = wav_label_dict

    # load json file
    def loadJson(self, json_path):
        with open(json_path, "rt", encoding= 'UTF-8') as f:
            j = json.load(f)
        return j

    # load wav file
    def loadWav(self, wav_path, sr, offset, duration):
        signal, sample_rate = librosa.load(wav_path, sr=sr, offset= offset, duration= duration)
        return signal, sample_rate

    # store sampling duration from json file
    def sampling(self, json_path):
        sample = []
        j = self.loadJson(json_path)
        for i in range(len(j["utterance"])):
            sample.append([j["utterance"][i]["start"], j["utterance"][i]["end"]])
        return sample

    # save spectrogram data as pickle
    def saveSpectrogram(self, spectrogram_data, pickle_name):
        save_base_dir = self.save_region_dir
        if not os.path.isdir(save_base_dir): os.mkdir(save_base_dir)

        save_pickle_path = os.path.join(save_base_dir, f"{pickle_name}_spectrogram.pickle")

        with open(save_pickle_path, "wb") as f:
            pickle.dump(spectrogram_data, f)

    def visualization(self, spectrogram_data, sample_rate, hop_length, file_name):
        save_image_dir = self.save_image_dir
        if not os.path.isdir(save_image_dir): 
            os.mkdir(save_image_dir)

        save_image_file = os.path.join(save_image_dir, f"{file_name}_spectrogram.png")

        plt.figure()
        librosa.display.specshow(spectrogram_data, sr= sample_rate, hop_length=hop_length)
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        plt.colorbar(format="%+2.0f dB")
        plt.title("Spectrogram (dB)")

        plt.savefig(save_image_file)



    def spectrogram(self, hop_length, n_fft, sr):
        wav_label_dict = self.wav_label_dict
        vis = self.vis
        vis_num = 1
        pickle_num = 1
        for i, (label_file, wav_file) in enumerate(wav_label_dict.items(), 1):
            label_path = os.path.join(self.label_file_path, label_file)
            wav_path = os.path.join(self.wav_file_path, wav_file)

            sample = self.sampling(label_path)
            for j in range(len(sample)):
                signal, sample_rate = self.loadWav(wav_path=wav_path, sr= sr, offset=sample[j][0], duration=(sample[j][1]-sample[j][0]))

                stft = librosa.stft(signal, n_fft= n_fft, hop_length= hop_length)
                magnitude = np.abs(stft)
                log_spectrogram = librosa.amplitude_to_db(magnitude)

                if log_spectrogram.shape[1]<401:
                    log_spectrogram = librosa.util.pad_center(log_spectrogram, size=401, axis=1)
                elif log_spectrogram.shape[1]>401 :
                    log_spectrogram = log_spectrogram[:, 0:401]

                self.saveSpectrogram(log_spectrogram, pickle_name= pickle_num)

                if vis_num < vis:
                    self.visualization(log_spectrogram, sample_rate=sr, hop_length=hop_length, file_name= vis_num)
                    vis_num +=1
                pickle_num +=1
