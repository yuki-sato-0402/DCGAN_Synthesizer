import os
import torch
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import soundfile as sf
from generator import Generator
from preProcess import AudioPreprocessor
from IPython.display import Audio, display

def inference_gan(preprocessor, time_frames):
    # Preparing the Generator Model
    generator = Generator(preprocessor)
    generator.load_state_dict(torch.load("model/generator.pth", map_location="cpu"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator.to(device)
    generator.eval()

    # Hyperparameters (same as VAE)
    n_noise = preprocessor.n_noise
    n_mels = preprocessor.n_mels
    sample_rate = preprocessor.sample_rate
    hop_length = preprocessor.hop_length
    n_fft = preprocessor.n_fft
    min_db = preprocessor.min_db
    max_db = preprocessor.max_db

    # Generating latent variables from noise and generating speech
    z = torch.randn(1, n_noise).to(device)
    # print("latent variable z:", z)

    # Generating a Mer Spectrogram with Generator
    with torch.no_grad():
        y = generator(z).cpu().numpy()

    # If the shape is [batch, channels, mel, time], use reshape.
    print("y.shape:", y.shape)
    image = y.reshape(n_mels, time_frames)
    print("image.shape:", image.shape)

    # Displaying the Mer Spectrogram
    plt.figure(figsize=(4, 4))
    librosa.display.specshow(image, sr=sample_rate, hop_length=hop_length, x_axis='time', y_axis='mel')
    plt.colorbar(format="%+2.0f dB")
    plt.title("Generated Mel-Spectrogram (GAN)")
    plt.show()

    # Converting a mel-spectrogram to audio (dB scale → power spectrum)
    mel_spec_db = image * (max_db - min_db) + min_db
    mel_spec = librosa.db_to_power(mel_spec_db)

    # Voice restoration by Griffin-Lim
    audio = librosa.feature.inverse.mel_to_audio(
        mel_spec,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_iter=32
    )

    audio = audio / (np.max(np.abs(audio)) + 1e-8)

    # play sound
    audio_obj = Audio(audio, rate=sample_rate)
    display(audio_obj)
    sf.write("outputAuido/generated_gan_audio.wav", audio, sample_rate)
    print("Audio has been saved.")

if __name__ == "__main__":


    targetSamplerate = 22050
    preprocessor = AudioPreprocessor(targetSamplerate )
    processed_mel_specs = preprocessor.process_single_file("60note_1.wav")  # Process a single file to determine time_frames
    print(processed_mel_specs[0].shape[1])  # Print the number of time frames in the mel-spectrogram
    
    inference_gan(preprocessor, processed_mel_specs[0].shape[1])

