#########################################################################################################
#
#   ELEC 475 - Lab 1
#   Fall 2025
#   Daniel Dubinko 
#   19dd34@queensu.ca

from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import torch
import interpolate
from model import autoencoderMLP4Layer
import argparse
from interpolate import Interpolate


# Load Model File
parser = argparse.ArgumentParser()
parser.add_argument("-l", "--load", type=str, help="Path to .pth file")
args = parser.parse_args()

if args.load:
    print(f"Loading file: {args.load}")
    file_path = f"{args.load}"
    print("File loaded successfully!\n")
else:
    print("No file provided with -l")
    file_path = 'MLP.8.pth'

# -----------------------------------------------------------------------
print("QUESTION 2:\n")

# Choose Image Index
idx = int(input("Enter index > "))

# Load MNIST Dataset
train_transform = transforms.Compose([transforms.ToTensor()])
train_set = MNIST('./data/mnist', train=True, download=True, transform=train_transform)

# Get image and label
image, label = train_set[idx]
plt.imshow(image.squeeze(), cmap='gray')
plt.show()

# -----------------------------------------------------------------------

print("\nQUESTION 4 (Testing Autoencoder):\n")
# Instantiate model and put it into evaluation mode 
model = autoencoderMLP4Layer()
# model.state_dict().keys() # depricated code, saving for future 
model.load_state_dict(torch.load(file_path, map_location='cpu'))
model.eval()

# Prepare image for model input by changing its shape 
model_input = image.reshape(1, 784)
print(f"Model input shape: {model_input.shape}")
print(f"Model input dtype: {model_input.dtype}\n")

# Run inference on image
with torch.no_grad():
    output = model(model_input)

# Reshape output for visualization
output_image = output.reshape(28, 28)

f = plt.figure()
f.add_subplot(1,2,1)
plt.imshow(image.squeeze(), cmap='gray')
f.add_subplot(1,2,2)
plt.imshow(output_image, cmap='gray')
plt.show()

# -----------------------------------------------------------------------

print("QUESTION 5 (Image Denoising):\n")
# Create Noise 
noise = torch.rand(1, 784)
# Add image and noise
noisy_image = model_input + noise

with torch.no_grad():
    denoised_image = model(noisy_image)

# Reshape images for visualization
noisy_image = noisy_image.reshape(28,28)
denoised_image = denoised_image.reshape(28, 28)

f = plt.figure()
f.add_subplot(1,3,1)
plt.imshow(image.squeeze(), cmap='gray')
f.add_subplot(1,3,2)
plt.imshow(noisy_image, cmap='gray')
f.add_subplot(1,3,3)
plt.imshow(denoised_image, cmap='gray')
plt.show()

# -----------------------------------------------------------------------

print("QUESTION 6 (Bottleneck Interpolation):\n")
# Choose two images
img_1_idx = int(input("Enter index of First Image > "))
img_2_idx = int(input("Enter index of Second Image > "))

image_1, label_1 = train_set[img_1_idx]
image_2, label_2 = train_set[img_2_idx]

# Print Labels
print(f"\nFirst image label: {label_1}")
print(f"Second image label: {label_2}")


# Return the bottleneck tensors for each image
image_1 = image_1.reshape(1, 784)
image_2 = image_2.reshape(1, 784)
encoded_img_1, encoded_img_2 = model.encode_two_images(image_1, image_2)

# Initialie Interpolation Module and Number of desired steps 
num_steps = 10
interpolate = Interpolate(image_1, image_2, num_steps, model=model)
interpolated_images = interpolate.apply()

# Create Figure and Plot the retrieved interpolated images
f = plt.figure()
i = 1

for img in interpolated_images:
    f.add_subplot(1,num_steps,i)
    plt.imshow(img, cmap='gray')
    i += 1
plt.show()

print("\nThank you for looking at my lab :) !!")





