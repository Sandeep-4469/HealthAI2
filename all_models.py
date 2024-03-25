import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.models import resnet50
from PIL import Image
import torch.nn.functional as F

labels = ['bone', 'lung', 'retina']

class CustomResNet(nn.Module):
    def __init__(self, num_classes):
        super(CustomResNet, self).__init__()
        # Load Pretrained ResNet50 model
        self.pretrained_model = resnet50(pretrained=True)
        
        # Freeze all layers except the final fully connected layer
        for param in self.pretrained_model.parameters():
            param.requires_grad = False
        
        # Modify the final fully connected layer
        in_features = self.pretrained_model.fc.in_features
        self.pretrained_model.fc = nn.Identity()  # Remove the final fully connected layer
        
        # Additional layer
        self.fc_additional = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU()
        )
        
        # Final fully connected layer
        self.fc_final = nn.Linear(128, num_classes)

    def forward(self, x):
        # Forward pass through the base model
        x = self.pretrained_model(x)
        
        # Additional layer
        x = self.fc_additional(x)
        
        # Final fully connected layer
        x = self.fc_final(x)
        
        return x

def xray_type(path):
    image = Image.open(path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    transformed_image = transform(image)
    transformed_image = transformed_image.unsqueeze(0)  # Add batch dimension
    
    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the trained model
    model = CustomResNet(3)
    model.load_state_dict(torch.load("xray.pt", map_location=device))
    model.eval()
    
    with torch.no_grad():
        outputs = model(transformed_image)
    
    probabilities = F.softmax(outputs, dim=1)
    _, predicted = torch.max(probabilities, 1)
    print("Predicted class:", labels[predicted.item()])
    return predicted.item()

# Define transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
