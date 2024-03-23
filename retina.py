from PIL import Image 
import torch
from torchvision import transforms

def retina(path):
    image = Image.open(path)
    model = torch.load("retina.pt", map_location=torch.device('cpu'))  # Load model on CPU
    model.eval()
    my_transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    if image.mode != 'RGB':
        image = image.convert('RGB')
    img = my_transform(image).unsqueeze(0)
    device = torch.device("cpu")  # Set device to CPU
    model = model.to(device)
    img = img.to(device)
    with torch.no_grad():
        score = model(img)
        print(score)
    min_index = torch.argmin(score)
    print(min_index)
    return min_index.item()  # Return the index as a Python scalar

print(retina("10003_left.jpeg"))
