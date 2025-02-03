from adversarial_image_generator.classifier import ResNetClassifier
from adversarial_image_generator.attack import AdversarialAttack
from adversarial_image_generator.preprocess import ImagePreprocessor
from adversarial_image_generator.utils import save_adversarial_image

def main():
    image_path = "car.png"  # Set your input image path here
    target_label = "carousel"  # Specify the target label (text)
    output_path = "adversarial_image.jpg"  # Output file for the adversarial image
    epsilon = 0.05  # Perturbation strength
    class_labels_file = "imagenet_classes.txt"  # Path to your class labels file

    # Initialize classes
    classifier = ResNetClassifier(class_labels_file)
    preprocessor = ImagePreprocessor()
    attack = AdversarialAttack(classifier.model, epsilon)

    # Convert target label to class index
    target_class_idx = classifier.label_to_index(target_label)

    # Preprocess the image
    image_tensor = preprocessor.preprocess(image_path)

    # Generate adversarial noise and apply it
    noise = attack.generate_noise(image_tensor, target_class_idx)
    adversarial_image = attack.apply_noise(image_tensor, noise)


    # Preprocess the image
    image_tensor = preprocessor.preprocess(image_path)

    # Predict original image
    print("Original Image Prediction:")
    predicted_label, confidence_score = classifier.predict(image_tensor)
    print(f"Predicted label: {predicted_label}, Confidence: {confidence_score:.4f}")

    # Generate adversarial noise and apply it
    noise = attack.generate_noise(image_tensor, target_class_idx)
    adversarial_image = attack.apply_noise(image_tensor, noise)

    # Predict the adversarial image
    print("Adversarial Image Prediction:")
    predicted_label, confidence_score = classifier.predict(adversarial_image)
    print(f"Predicted label: {predicted_label}, Confidence: {confidence_score:.4f}")

    # Save the adversarial image
    save_adversarial_image(adversarial_image, output_path)

if __name__ == "__main__":
    main()
