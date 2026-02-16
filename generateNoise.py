import matplotlib.pyplot as plt
import numpy as np
import torch


#Generate and display images
def generate_images(generator, preprocessor, device):

    n_rows = 4
    n_cols = 4
    noise = torch.randn(n_rows * n_cols, preprocessor.n_noise).to(device)
    g_imgs = generator(noise)
    g_imgs = g_imgs/2 + 0.5  # Set to a range of 0-1
    g_imgs = g_imgs.cpu().detach().numpy()  # Convert to numpy array for visualization

    time_frames_spaced = preprocessor.time_frames + 2
    n_mels_spaced = preprocessor.n_mels + 2

    # Overall image
    matrix_image = np.zeros((time_frames_spaced*n_rows, n_mels_spaced*n_cols))

    #  Arrange the generated images side by side to create a single image.
    for r in range(n_rows):
        for c in range(n_cols):
          # Convert to time × frequency resolution
            g_img = g_imgs[r*n_cols + c].reshape(preprocessor.time_frames, preprocessor.n_mels)
            top = r*time_frames_spaced
            left = c*n_mels_spaced
            matrix_image[top : top+preprocessor.time_frames, left : left+preprocessor.n_mels] = g_img

    plt.figure(figsize=(8, 8))
    plt.imshow(matrix_image, cmap="Greys_r", vmin=0.0, vmax=1.0)
    plt.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)  # Erase axis labels and lines
    plt.show()


# Calculation of correct answers
def count_correct(y, t):
    correct = torch.sum((torch.where(y<0.5, 0, 1) ==  t).float())
    return correct.item()