from PIL import Image
import torch
from torchvision import transforms

class ImagePreprocessor:
    def __init__(self, target_size=(224, 224)):
        self.transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def preprocess(self, image_path):
        """Load and preprocess the image."""
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0)  # Add batch dimension
        return image_tensor
