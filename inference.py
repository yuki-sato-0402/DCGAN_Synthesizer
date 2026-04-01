import os
import torch
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import soundfile as sf
import argparse
from generator import Generator
from preProcess import AudioPreprocessor
from IPython.display import Audio, display

def inference_gan(preprocessor):
    # Preparing the Generator Model
    generator = Generator(preprocessor)
    
    if preprocessor.mode == 'mel':
        model_path = "model/generatorMel.pth"
    else:
        model_path = "model/generatorSTFT.pth"
        
    if os.path.exists(model_path):
        generator.load_state_dict(torch.load(model_path, map_location="cpu"))
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator.to(device)
    generator.eval()

    # Hyperparameters
    n_noise = preprocessor.n_noise
    sample_rate = preprocessor.sample_rate
    hop_length = preprocessor.hop_length
    n_fft = preprocessor.n_fft
    min_db = preprocessor.min_db
    max_db = preprocessor.max_db

    # Generating latent variables from noise
    z = torch.randn(1, n_noise).to(device)

    # Generating with Generator
    with torch.no_grad():
        # shape [1, channels, freq, time]
        y = generator(z).cpu().numpy()

    if preprocessor.mode == 'mel':
        # Mel mode
        mel_norm = y[0, 0, :, :]
        # Denormalize [-1, 1] -> dB
        mel_db = (mel_norm + 1) / 2 * (max_db - min_db) + min_db
        # Power to linear
        mel_power = librosa.db_to_power(mel_db)
        
        # Restoration using Griffin-Lim
        audio = librosa.feature.inverse.mel_to_audio(
            mel_power, sr=sample_rate, n_fft=n_fft, hop_length=hop_length
        )
        
        magnitude_for_viz = librosa.power_to_db(mel_power, ref=np.max)
        title = 'Generated Mel Spectrogram'
        
        
        audio = audio / (np.max(np.abs(audio)) + 1e-8)
        
        # play sound
        # audio_obj = Audio(audio, rate=sample_rate)
        # display(audio_obj)
        sf.write("outputAuido/generated_gan_audioMel.wav", audio, sample_rate)
        print("Audio has been saved.")
    else:
        # STFT mode
        mag_norm = y[0, 0, :, :]
        phase_diff_norm = y[0, 1, :, :]
        
        # 2. Amplitude Restoration
        mag_db = (mag_norm + 1) / 2 * (max_db - min_db) + min_db
        magnitude = librosa.db_to_amplitude(mag_db)
        
        # 3. Phase Restoration
        phase_diff = phase_diff_norm * np.pi
        phase = np.cumsum(phase_diff, axis=1)
        
        # 4. Reconstruction of Complex STFT
        stft_reconstructed = magnitude * (np.cos(phase) + 1j * np.sin(phase))
        
        # 6. Speech Restoration Using iSTFT
        audio = librosa.istft(
                stft_reconstructed,
                hop_length=hop_length,
                n_fft=n_fft
            )
        magnitude_for_viz = librosa.amplitude_to_db(magnitude, ref=np.max)
        title = 'Generated STFT Magnitude'
        
        audio = audio / (np.max(np.abs(audio)) + 1e-8)
        
        # play sound
        # audio_obj = Audio(audio, rate=sample_rate)
        # display(audio_obj)
        sf.write("outputAuido/generated_gan_audioSTFT.wav", audio, sample_rate)

    # 5. Visualization
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(magnitude_for_viz,
                             sr=sample_rate, hop_length=hop_length, y_axis='mel' if preprocessor.mode == 'mel' else 'linear', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="stft", choices=["stft", "mel"], help="Preprocessing mode")
    parser.add_argument("--file", type=str, default="60note_1.wav", help="Input file for sizing")
    args = parser.parse_args()

    targetSamplerate = 22050
    preprocessor = AudioPreprocessor(targetSamplerate, mode=args.mode)
    # Note: process_single_file might be needed to set some internal states if any, 
    # but here we use it to match the logic in the original script.
    _ = preprocessor.process_single_file(args.file)
    
    inference_gan(preprocessor)
