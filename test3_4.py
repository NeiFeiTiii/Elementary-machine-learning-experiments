import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# Define the neural network model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load the saved model weights
model = Net()
model.load_state_dict(torch.load('mnist_model.pth'))
model.eval()

# Define image preprocessing steps
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Directory containing the images
image_dir = 'imageDataSet'

# List to store images and predictions
images, preds = [], []

# Load, preprocess, and predict images
for image_name in os.listdir(image_dir):
    image_path = os.path.join(image_dir, image_name)
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)

    images.append(image.squeeze(0))  # Remove batch dimension
    preds.append(predicted.item())

# Visualize the images and predictions
fig, axes = plt.subplots(1, len(images), figsize=(15, 1.5))
for i, image in enumerate(images):
    img = image.numpy().squeeze()
    axes[i].imshow(img, cmap='gray')
    axes[i].set_title(f'Pred: {preds[i]}')
    axes[i].axis('off')
plt.show()