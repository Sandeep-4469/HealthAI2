import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

def plot_prediction(image, prediction, output_path):
    image = image.squeeze().permute(1, 2, 0).numpy() * 255  # Convert to NumPy array and scale to [0, 255]
    image = image.astype('uint8')  # Convert to uint8
    pil_image = Image.fromarray(image)  # Convert NumPy array to PIL image
    draw = ImageDraw.Draw(pil_image)  # Create a draw object
    classes=['elbow positive', 'fingers positive', 'forearm fracture', 'humerus fracture', 'humerus', 'shoulder fracture', 'wrist positive']
    boxes = prediction[0]['boxes'].numpy()
    scores = prediction[0]['scores'].numpy()
    labels = prediction[0]['labels'].numpy()

    for box, score, label in zip(boxes, scores,labels):
        if score > 0.6:  # Adjust the threshold based on your needs
            box = [round(coord, 2) for coord in box]
            draw.rectangle([box[0], box[1], box[2], box[3]], outline='red',width=10)  # Draw bounding box
            draw.text((box[0], box[1]), f'{label}', fill='red',font_size=60)  # Add text

    # Save the image with bounding boxes
    pil_image.save(output_path)
    return pil_image

def bone_main(image_path):
    img = Image.open(image_path)
    # Check if CUDA is available, if not, set device to CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the pre-trained Faster R-CNN model with 91 classes
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # Modify the classifier head for 7 classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=7)

    # Load the state dictionary
    loaded_model_path = 'Bone.pt'
    state_dict = torch.load(loaded_model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    preprocess = transforms.Compose([
        transforms.ToTensor(),
    ])
    input_image = preprocess(img.convert('RGB')).unsqueeze(0)

    # Move the input image to the device
    input_image = input_image.to(device)

    # Make the prediction
    with torch.no_grad():
        predictions = model(input_image)

    # Move predictions to CPU if they are on GPU
    predictions = [{k: v.to('cpu') for k, v in t.items()} for t in predictions]
    output_image_path = 'static/output_image.jpg'
    plot_prediction(input_image.squeeze(), predictions, output_image_path)
    print(f"Image with bounding boxes saved to {output_image_path}")


