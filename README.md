# Adversarial Image Generator

This project generates adversarial images by applying adversarial noise to input images with intended label, tricking a deep learning classifier into misclassifying the orginal class to the target class.

## Features
- Load and classify an image using a ResNet classifier.
- Generate adversarial noise to mislead the classifier.
- Apply the noise and generate an adversarial image.
- Predict and compare classifications before and after the attack.

## Installation
Ensure you have Python installed, then install dependencies:

```sh
pip install -r requirements.txt
```

## Usage
Run the following command to generate an adversarial image:

```sh
python generate_adversarial.py
```

This will:
1. Classify the original image and display the predicted label with confidence.
2. Generate adversarial noise and apply it to the image.
3. Classify the adversarial image and display the new predicted label with confidence.
4. Save the adversarial image as `adversarial_image.jpg`.

## Configuration
Modify the parameters in `generate_adversarial.py`:
- **`image_path`**: Path to the input image.
- **`target_label`**: The desired misclassification label.
- **`output_path`**: Where to save the adversarial image.
- **`epsilon`**: Strength of the perturbation.

## File Structure
```
.
├── adversarial_image_generator/
│   ├── classifier.py
│   ├── attack.py
│   ├── preprocess.py
│   ├── utils.py
├── generate_adversarial.py
├── imagenet_classes.txt
├── requirements.txt
├── README.md
```

## Example Output
```
Original Image Prediction:
Predicted label: sports_car, Confidence: 0.5380

Adversarial Image Prediction:
Predicted label: carousel, Confidence: 0.1204

Adversarial image saved to adversarial_image.jpg
```


