from torchvision import transforms

def save_adversarial_image(image_tensor, output_path):
    image = transforms.ToPILImage()(image_tensor.squeeze(0))  # Remove batch dimension
    image.save(output_path)
    print(f"Adversarial image saved to {output_path}")
