import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
class CnnModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 100, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(100, 150, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 

            nn.Conv2d(150, 200, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(200, 200, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 

            nn.Conv2d(200, 250, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(250, 250, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 

            nn.Flatten(), 
            nn.Linear(36000, 64),  
            nn.ReLU(),            
            nn.Linear(64, 32),  
            nn.ReLU(),            
            nn.Linear(32, 16),           
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(8, 4)
        )
        
    def forward(self, xb):
        return self.network(xb)

# Instantiate the model
loaded_model = CnnModel()

# Move the model to CPU
device = torch.device('cpu')
loaded_model.load_state_dict(torch.load('lung.pt', map_location=device))
loaded_model.eval()

# Define transformation
train_transform = transforms.Compose([
    transforms.Resize((100, 100)),  # Resize the image
    transforms.ToTensor(),          # Convert the PIL image to a PyTorch tensor
])

def test_lung(image_path):
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')

    transformed_image = train_transform(image)
    numpy_array = transformed_image.permute(1, 2, 0).numpy()
    pil_image = Image.fromarray((numpy_array * 255).astype(np.uint8))
    pil_image.save('static/output_image.jpg')
    transformed_image = transformed_image.unsqueeze(0)  # Add batch dimension

    # Forward pass
    with torch.no_grad():
        output = loaded_model(transformed_image)

    # Compute softmax probabilities
    probabilities = F.softmax(output, dim=1)

    # Convert output probabilities to predicted class
    _, predicted = torch.max(probabilities, 1)

    # Print the predicted class and probabilities for each class
    class_names = ['PNEUMONIA', 'NORMAL', 'COVID19', 'TURBERCULOSIS']
    d = {}
    d["xray"] = "lung"
    for i, prob in enumerate(probabilities.squeeze().tolist()):
        d[class_names[i]] = prob 
    return d
