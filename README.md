# DCGAN_Synthesizer
DCGAN-based synthesizer using GM Sound Source as learning material

This project is still in the experimental stage, and I plan to add new features and improve model accuracy in the future.

## Folder Structure

- `inputAudio/` – It contains audio files of the study data.
- `outputAudio/` – This contains audio files of the output results from model inference.
- `model/` – This is the folder where the trained models are output. generator outputs two types of files, pth and onnx, while discriminator outputs only the pth file. Only the generator is required for model inference.

## How to Use

Just open the notebook(s) in Google Colab and run all the cells from top to bottom.  
No need to install anything locally — everything runs in the cloud.  
Each step is explained in the notebook itself.

## About Training Data
In this project, I have collected about 110 types of GM (Fluid synth) C4 (note 60) sound sources. Information about Fluid Synth can be found [here](https://pypi.org/project/pyfluidsynth/).