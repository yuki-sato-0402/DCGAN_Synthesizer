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
    model_path = "model/generator.pth"
    if os.path.exists(model_path):
        generator.load_state_dict(torch.load(model_path, map_location="cpu"))
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator.to(device)
    generator.eval()

    # Hyperparameters (same as VAE)
    n_noise = preprocessor.n_noise
    #n_mels = preprocessor.n_mels
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
        # shape [1, 2, freq, time]
        y = generator(z).cpu().numpy()

    # 1. Channel separation
    # y[0, 0] is amplitude (mag), y[0, 1] is phase difference (phase_diff)
    mag_norm = y[0, 0, :, :]
    phase_diff_norm = y[0, 1, :, :]
    
    # 2. Amplitude Restoration (Denormalization [-1, 1] → dB → Linear)
    mag_db = (mag_norm + 1) / 2 * (max_db - min_db) + min_db
    magnitude = librosa.db_to_amplitude(mag_db)
    
    # 3. Phase Restoration (Denormalization [-1, 1] → Phase Difference)
    phase_diff = phase_diff_norm * np.pi  # Scale back to [-π, π]
    
    #Accumulate phase difference in the frame (time axis) direction to restore phase
    phase = np.cumsum(phase_diff, axis=1)
    
    # 4. Reconstruction of Complex STFT
    real = magnitude * np.cos(phase)
    imag = magnitude * np.sin(phase)
    #It is customary to use j rather than i for the imaginary unit.
    stft_reconstructed = real + 1j * imag  # Convert to a complex array
    
    # 5. Visualization (Display amplitude components)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.amplitude_to_db(magnitude, ref=np.max),
                             sr=sample_rate, hop_length=hop_length, y_axis='mel', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Generated STFT Magnitude')
    plt.tight_layout()
    plt.show()
    

    # 6. Speech Restoration Using iSTFT (Griffin-Lim Not Required)
    audio = librosa.istft(
            stft_reconstructed,
            hop_length=hop_length,
            n_fft=n_fft
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
    processed_stft = preprocessor.process_single_file("60note_1.wav")  # Process a single file to determine time_frames
    print(processed_stft[0].shape[1])  # Print the number of time frames in the stft
    inference_gan(preprocessor, processed_stft[0].shape[1])


