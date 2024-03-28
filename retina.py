from PIL import Image 
import torch
from torchvision import transforms
import numpy as np
def test_retina(path):
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
    img = my_transform(image)
    numpy_array = img.permute(1, 2, 0).numpy()
    pil_image = Image.fromarray((numpy_array * 255).astype(np.uint8))
    pil_image.save('static/output_image.jpg')
    img = img.unsqueeze(0)
    device = torch.device("cpu")  # Set device to CPU
    model = model.to(device)
    img = img.to(device)
    with torch.no_grad():
        score = model(img)
        print(score)
    labels = ["NO DR","MIND DR","MODERATE DR","SEVERE DR","PROLIFERATE DR"]
    min_index = torch.argmin(score)
    print(min_index)
    d = {}
    d["xray"] = "retina"
    d["diabetic retinopathy"] = labels[min_index.item()]
    return d

