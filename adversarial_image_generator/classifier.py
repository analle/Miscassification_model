import torch
import torch.nn.functional as F
from torchvision import models

class ResNetClassifier:
    def __init__(self, class_labels_file):
        self.model = models.resnet18(pretrained=True)
        self.model.eval()  # Set the model to evaluation mode
        self.class_labels = self.load_class_labels(class_labels_file)

    def load_class_labels(self, file_path):
        """Load class labels from a comma-separated file."""
        class_labels = {}
        with open(file_path, 'r') as f:
            for line in f:
                index, label = line.strip().split(",")
                class_labels[int(index)] = label.strip()
        return class_labels

    def predict(self, image_tensor):
        """Get the predicted label and confidence score for an image tensor."""
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probs = F.softmax(outputs, dim=1)  # Softmax for probabilities
            confidence, predicted_idx = torch.max(probs, 1)
            predicted_label = self.class_labels[predicted_idx.item()]
            confidence_score = confidence.item()
        
        return predicted_label, confidence_score

    def label_to_index(self, label):
        """Convert a label to the corresponding class index."""
        for idx, lbl in self.class_labels.items():
            if lbl == label:
                return idx
        raise ValueError(f"Label '{label}' not found in the class labels.")
