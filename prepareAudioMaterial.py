import os
import numpy as np
import pretty_midi
from scipy.signal import resample
from scipy.io.wavfile import write as wav_write


def export_all_instruments(Target_Samplerate):
    target_samplerate = Target_Samplerate
    target_length = target_samplerate * 2
    dir_path = "trainingData"

    for program in range(1, 89):  # Program No. 1-88

        # generating midi
        music = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=program)
        note = pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=2.0)
        instrument.notes.append(note)
        music.instruments.append(instrument)

        # synthesize with SoundFont
        audio_data = music.fluidsynth(sf2_path="Touhou.sf2")

        # resampling
        n_samples = int(len(audio_data) * target_samplerate / 44100)
        # print(len(audio_data))
        audio_data_resampled = resample(audio_data, n_samples)
        # print(len(audio_data_resampled))

        # cuting out only the required length
        if len(audio_data_resampled) < target_length:
            print(f"⚠️ Skipped {program} (too short: {len(audio_data_resampled)} samples)")
            continue

        clipped = audio_data_resampled[:target_length]

        # export
        filename = os.path.join(dir_path, f"60note_{program}.wav")
        if not os.path.exists(filename):
            wav_write(filename, target_samplerate, (clipped * 32767).astype(np.int16))
            print(f"✔️ {filename} Exported ({target_length} samples)")
        else:
            print(f"⏭️ {filename} already exists.")