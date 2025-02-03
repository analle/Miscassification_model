import torch
import torch.optim as optim
import torch.nn.functional as F

class AdversarialAttack:
    def __init__(self, model, epsilon=0.05, max_epochs=1000):
        self.model = model
        self.epsilon = epsilon
        self.max_epochs = max_epochs

    def generate_noise(self, image_tensor, target_class_idx):
        """Generate adversarial noise using FGSM."""
        noise = torch.zeros_like(image_tensor, requires_grad=True)
        optimizer = optim.Adam([noise], lr=0.01)

        for epoch in range(self.max_epochs):
            perturbed_image = torch.clamp(image_tensor + noise, 0, 1)
            outputs = self.model(perturbed_image)
            predicted_class = torch.argmax(outputs, dim=1).item()

            if predicted_class == target_class_idx:
                break

            loss = F.cross_entropy(outputs, torch.tensor([target_class_idx]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            noise.data = torch.clamp(noise.data, -self.epsilon, self.epsilon)

        return noise

    def apply_noise(self, image_tensor, noise):
        """Apply the generated adversarial noise to the image."""
        return torch.clamp(image_tensor + noise, 0, 1)
