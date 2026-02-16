import os
import torch
import torch.nn as nn
import torch.optim as optim
from prepareAudioMaterial import export_all_instruments
from preProcess import AudioPreprocessor
from generator import Generator
from discriminator import Discriminator
from generateNoise import generate_images, count_correct

def trainModel(preprocessor, generator, discriminator, train_loader, device):
    # Binary cross-entropy error function
    loss_func = nn.BCELoss()

    # This is a hyperparameter of the Adam optimizer that specifies the decay rate of the exponential moving average.
    optimizer_gen = optim.Adam(generator.parameters(), lr=0.01, betas=(0.5, 0.999))
    optimizer_disc = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))

    error_record_fake = []  # Fake image error log
    acc_record_fake = []  # Fake image accuracy record
    error_record_real = []  # Authentic image error log
    acc_record_real = []  # Accuracy record of authentic images

    # Training
    generator.train()
    discriminator.train()
    for i in range(preprocessor.epochs):
        loss_fake = 0  # 誤差
        correct_fake = 0  # 正解数
        loss_real = 0
        correct_real = 0
        n_total = 0  # Total number of data (used for accuracy calculation)
        for j, (x,) in enumerate(train_loader):  # Extract mini batch (x,)

            n_total += x.size()[0]  # Cumulative batch size

            # Generate images from noise and train the discriminator
            noise = torch.randn(x.size()[0], preprocessor.n_noise).to(device)
            imgs_fake = generator(noise)  # Image generation
            t = torch.zeros(x.size()[0], 1).to(device)  # The correct answer is 0.

            #print(f"Input shape Generator: {noise.shape}")
            y = discriminator(imgs_fake)

            #print("y ", y.shape)
            loss = loss_func(y, t)
            optimizer_disc.zero_grad()
            loss.backward()
            optimizer_disc.step()  # Update only discriminator parameters
            loss_fake += loss.item()
            correct_fake += count_correct(y, t)

            # Train the discriminator using real images.
            imgs_real= x.to(device)
            t = torch.ones(x.size()[0], 1).to(device) # The correct answer is 1.

            # print("second")
            y = discriminator(imgs_real)
            loss = loss_func(y, t)
            optimizer_disc.zero_grad()
            loss.backward()
            optimizer_disc.step()  # Update only discriminator parameters
            loss_real += loss.item()
            correct_real += count_correct(y, t)

            # Train the generator
            noise = torch.randn(x.size()[0]*2, preprocessor.n_noise).to(device)  # Double the batch size
            imgs_fake = generator(noise)  # Image generation
            t = torch.full((x.size()[0]*2, 1), 0.9).to(device)  # It looks real, but it's a little vague.
            # print("third")
            y = discriminator(imgs_fake)
            loss = loss_func(y, t)
            optimizer_gen.zero_grad()
            loss.backward()
            optimizer_gen.step()  # Update only generator parameters

        loss_fake /= j+1  # error
        error_record_fake.append(loss_fake)
        acc_fake = correct_fake / n_total  # accuracy
        acc_record_fake.append(acc_fake)

        loss_real /= j+1  # error
        error_record_real.append(loss_real)
        acc_real = correct_real / n_total  # accuracy
        acc_record_real.append(acc_real)

        # Display errors, accuracy, and generated images at regular intervals.
        if i % 10 == 0:
            print ("Epochs:", i)
            # Discriminator error and accuracy for generated images (fakes)
            print ("Error_fake:", loss_fake , "Acc_fake:", acc_fake)
            # Discriminator error and accuracy for real images
            print ("Error_real:", loss_real , "Acc_real:", acc_real)
            #generate_images(generator, preprocessor, device)

    # Save trained model
    torch.save(generator.state_dict(), "model/generator.pth")
    torch.save(discriminator.state_dict(), "model/discriminator.pth")

    # export as onnx model
    generator.eval()
    dummy_input = torch.randn(1, preprocessor.n_noise).to(device)
    torch.onnx.export(generator,
                    dummy_input,
                    "model/generator.onnx",
                    export_params=True,
                    opset_version=10,
                    do_constant_folding=True, # Constant folding optimization
                    input_names = ['input'],
                    output_names = ['output'],
                    dynamic_axes={'input' : {0 : 'batch_size'}, 'output' : {0 : 'batch_size'}})
    print("The model has been saved.")


if __name__ == "__main__":
    
    
    targetSamplerate = 22050

    export_all_instruments(targetSamplerate)

    preprocessor = AudioPreprocessor(targetSamplerate )
    train_loader, n_in_out = preprocessor.get_trainloder()

    generator = Generator(preprocessor)  # preprocessor is passed during training

    # Check for GPU availability and use CPU if not available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator.to(device)
    print(f"Using device: {device}")
    #print(generator)

    discriminator = Discriminator(preprocessor)
    discriminator.to(device)
    #print(discriminator)

    trainModel(preprocessor, generator, discriminator, train_loader, device)