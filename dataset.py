from tqdm import tqdm
import numpy as np
import torchaudio as ta
import os
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import utils
import spaudiopy as spa
import torch
from natsort import natsorted
import soundfile as sf

def load_audio(path):
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim == 1:
        audio = audio[:, None]
    audio = torch.from_numpy(audio).T  # [T,C] -> [C,T]
    return audio, sr

class AudioDataset(Dataset):
    def __init__(self,
                 chunk_size_ms=2000,
                 overlap=0.5,
                 folder_path=None,
                 device='cuda',):
        super().__init__()
        self.mic_array_audio = []
        self.ambisonic_audio = []
        self.wav_files_foa = []
        self.wav_files_mic = []
        self.folder_path = folder_path
        self.chunk_size = chunk_size_ms * 24
        self.overlap = overlap
        self.chunks = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


        pbar = tqdm(range(15))
        for subject_id in pbar:
            pbar.set_description(f"loading data: subject {subject_id + 1}")
            mic_folder_path = f"/data/huangcy/Parameters_Ec/ML_Method/data/data_generated_sorted/{subject_id + 1}"
            foa_folder_path = f"/data/huangcy/Parameters_Ec/ML_Method/data/data_generated_FOA_sorted/{subject_id +1}"
            mic_subject_audio = []
            foa_subject_audio = []

            # os.makedirs(os.path.join("data", "data_generated_sorted", f"{subject_id + 1}"), exist_ok=True)
            # os.makedirs(os.path.join("data", "data_generated_FOA_sorted", f"{subject_id + 1}"), exist_ok=True)
            # wav_id = 0
            for mic_file, foa_file in zip(natsorted(os.listdir(mic_folder_path)),natsorted(os.listdir(foa_folder_path))):
                mic_file_path = os.path.join(mic_folder_path, mic_file)
                foa_file_path = os.path.join(foa_folder_path, foa_file)
                print(mic_file_path)
                print(foa_file_path)
                #
                # mic_path = os.path.join(f"./data/data_generated_sorted/{subject_id + 1}", f"circular_{wav_id}.wav" )
                # foa_path = os.path.join(f"./data/data_generated_FOA_sorted/{subject_id + 1}", f"FOA_{wav_id}.wav")

                mic_audio, fm = load_audio(mic_file_path)
                foa_audio, ff = load_audio(foa_file_path)
                
                
                if mic_audio.shape[1] <= 24000 * 2:
                    continue
                

                # wav_id += 1

                mic_audio = mic_audio
                foa_audio = foa_audio

                mic_subject_audio.append(mic_audio)
            self.mic_array_audio.append(mic_subject_audio)



            for foa_file in natsorted(os.listdir(foa_folder_path)):
                foa_file_path = os.path.join(foa_folder_path, foa_file)
                print(foa_file_path)
                foa_audio, _ = load_audio(foa_file_path)
                
                if foa_audio.shape[1] <= 24000 * 2:
                    continue
                foa_audio = foa_audio
                foa_subject_audio.append(foa_audio)
            self.ambisonic_audio.append(foa_subject_audio)


        for subject_id in range(5):
            wave_id = 0
            for mic_audio in self.mic_array_audio[subject_id]:
                last_chunk_start_frame = mic_audio.shape[-1] - self.chunk_size + 1
                hop_length = int((1 - overlap) * self.chunk_size)
                for offset in range(0, last_chunk_start_frame, hop_length):
                    self.chunks.append({'subject': subject_id,'wave_id': wave_id, 'offset': offset})
                wave_id += 1

    def __len__(self):
        """
        :return:number of training chunks in dataset
        """
        return len(self.chunks)

    def __getitem__(self, index):
        subject = self.chunks[index]['subject']
        wave_id = self.chunks[index]['wave_id']
        offset = self.chunks[index]['offset']

        mic_audio = self.mic_array_audio[subject][wave_id][:, offset:offset+self.chunk_size]
        foa_audio = self.ambisonic_audio[subject][wave_id][:, offset:offset+self.chunk_size]
        #print(self.mic_array_audio[subject][wave_id].shape)
        #print(self.ambisonic_audio[subject][wave_id].shape)
        

        # print(f"subject {subject} wave_id {wave_id} offset {offset}")

        return mic_audio, foa_audio




        
        












