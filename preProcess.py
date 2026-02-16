import os
import numpy as np
import librosa
import torch
from torch.utils.data import Sampler
from torch.nn.utils.rnn import pad_sequence
import random


class PaddedBatchSampler(Sampler):
    #Sampler that supplements the last batch to the specified size
    def __init__(self, dataset_size, batch_size):
        self.dataset_size = dataset_size
        self.batch_size = batch_size

    def __iter__(self):
        #Create an index list
        indices = list(range(self.dataset_size))

        # Calculate the number of samples needed to complete the last batch.
        remainder = self.dataset_size % self.batch_size
        padding_size = 0 if remainder == 0 else self.batch_size - remainder

        # Fill in the gaps by randomly selecting from existing data.
        if padding_size > 0:
            padding_indices = random.choices(indices, k=padding_size)
            indices = indices + padding_indices

        # shuffle the index
        random.shuffle(indices)
        return iter(indices)

    def __len__(self):
        # Returns the number of samples divisible by the batch size.
        padding_size = 0 if self.dataset_size % self.batch_size == 0 else self.batch_size - (self.dataset_size % self.batch_size)
        return self.dataset_size + padding_size

class AudioPreprocessor:
    def __init__(self, Target_Samplerate):
        self.sample_rate = Target_Samplerate
        self.segment_length = 2 * self.sample_rate  # Number of samples for 2 seconds
        self.min_length = int(1.5 * self.sample_rate)  # Number of samples for 1.5 seconds
        #self.n_mels = 128  #Height of output of melspectrogram (number of dimensions)
        self.hop_length = 512

        # The number of noise "How diverse the output can be" affects the number of dimensions in the latent space.
        self.n_noise = 256 #This will be the input value for generater.
        self.epochs = 250
        self.n_fft = 1024
        self.min_db = -80.0
        self.max_db = 0.0
        
        self.time_frames = None  # This will be set after processing the data to determine the number of time frames in the mel-spectrogram.

    def load_audio(self, file_name):
       #Load audio file
        file_path = os.path.join(self.audio_path, file_name)
        audio, _ = librosa.load(file_path, sr=self.sample_rate)
        return audio

    def adjust_segment(self, audio):
        #Divide the audio data into 2-second segments and zero-pad as necessary.
        segments = []
        for i in range(0, len(audio), self.segment_length):
            segment = audio[i:i + self.segment_length]
            if len(segment) == self.segment_length:
                segments.append(segment)
            elif len(segment) >= self.min_length:
                # Zero padding for values greater than 1.5 seconds
                padded_segment = np.pad(segment, (0, self.segment_length - len(segment)), mode='constant')
                segments.append(padded_segment)
        return segments

    #def preprocess_audio(self, audio):
    #    #Converting audio data to melspectrograms
    #    mel_spec = librosa.feature.melspectrogram(
    #        y=audio, sr=self.sample_rate, n_mels=self.n_mels, hop_length=self.hop_length, n_fft=self.n_fft
    #    )
    #    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    #    mel_spec_db = np.clip(mel_spec_db, self.min_db, self.max_db)
    #    mel_spec_db = 2 * (mel_spec_db - self.min_db) / (self.max_db - self.min_db) - 1
    #    print(f"Mel-spectrogram shape: {mel_spec_db.shape}")#Mel-spectrogram shape: (128, 87)
    #    return mel_spec_db
    def preprocess_audio(self, audio):
        # 1. Execute STFT (complex64)
        stft = librosa.stft(y=audio, n_fft=self.n_fft, hop_length=self.hop_length)
        
        # 2. Separation into amplitude and phase
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # 3. Calculation of Phase Difference (Instantaneous Frequency)
        # Pad the beginning of the column with 0 or the initial phase to take the difference from the first frame
        phase_diff = np.diff(phase, axis=1, prepend=phase[:, :1])
        
        # 4. Phase wrapping (-π to π)
        phase_diff_wrapped = (phase_diff + np.pi) % (2 * np.pi) - np.pi
        
        # 5.Amplitude compression (logarithmic scaling is common)
        mag_db = librosa.amplitude_to_db(magnitude, ref=np.max)
        # Normalization (-1 to 1)
        mag_norm = 2 * (mag_db - self.min_db) / (self.max_db - self.min_db) - 1
        print(f"mag_norm shape: {mag_norm.shape}")
        
        phase_norm = phase_diff_wrapped / np.pi  # Scale to [-1, 1]
        print(f"phase_norm shape: {phase_norm.shape}")

        # Stack the amplitude and phase to form a (2, Freq, Time) shape
        combined = np.stack([mag_norm, phase_norm], axis=0)
        print(f"combined shape: {combined.shape}") #(2, 1025, 87)
        return combined

    def process_all_files(self, audio_path):
        #Process all audio files in the directory and return a stft.
        self.audio_path = audio_path
        all_stft = []
        for file_name in os.listdir(audio_path):
            if file_name.endswith('.wav'):
                print(f"Processing {file_name}...")
                audio = self.load_audio(file_name)
                segments = self.adjust_segment(audio)
                stft = [self.preprocess_audio(segment) for segment in segments]
                all_stft.extend(stft)
        print("All files processed.")
        return all_stft
    
    def process_single_file(self, file_name):
        #Process a single audio file and return a mel-spectrogram.
        self.audio_path = "trainingData"
        audio = self.load_audio(file_name)
        segments = self.adjust_segment(audio)
        stft = [self.preprocess_audio(segment) for segment in segments]
        print("File processed.")
        return stft

    def collate_fn(self, batch):
        # batch is a list of tuples (stft)
        stfts = [item[0] for item in batch]

        # Pad stfts to the maximum length in the batch
        stfts_padded = pad_sequence(stfts, batch_first=True)
        return (stfts_padded,)

    def create_dataloader(self, stft_list, batch_size=16):
        #Create a data loader from stft (complete the last batch as well)
        stft_array = np.array(stft_list)
        stft_tensor = torch.tensor(stft_array, dtype=torch.float)
        dataset = torch.utils.data.TensorDataset(stft_tensor)

        # Use a custom sampler to fill in the gaps.
        sampler = PaddedBatchSampler(len(dataset), batch_size)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,  # If you specify a sampler, shuffle will be ignored.
            collate_fn=self.collate_fn
        )

        #Calculate how many batches there will be in the last batch based on the dataset size and batch size.
        num_samples = len(dataset)
        num_complete_batches = num_samples // batch_size
        remaining_samples = num_samples % batch_size

        print(f"dataset size: {num_samples}")
        print(f"total number of batches: {num_complete_batches}")
        if remaining_samples > 0:
            print(f"Number of samples in the last batch: {remaining_samples} (Complemented to {batch_size})")
        else:
            print("All batches are complete.")

        return dataloader

    def get_trainloder(self):
        self.audio_path = "trainingData"

        stft = self.process_all_files(self.audio_path)
        print(f"Number of processed stfts: {len(stft)}")

        # Create a data loader (with completion feature)
        dataloader = self.create_dataloader(stft, batch_size=16)


        #This is a tuple (tensor) ←tuple with 1 element.
        for batch in dataloader:
            x = batch[0]
            total_elements = x.shape[1] * x.shape[2]  # 128 * 87 = 11136
            break


        self.time_frames = x.shape[2]
        print(f"Batch shape : {x.shape}")

        print(f"Total number of elements in a batch: {total_elements}")

        return dataloader, total_elements

