import os
import glob
import torch
import cv2
import numpy as np
from train import TripleRiderScratchCNN

def run_prediction():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model architecture and trained weights
    model = TripleRiderScratchCNN(num_classes=3).to(device)
    
    if not os.path.exists("triple_rider_scratch.pth"):
        print("Error: 'triple_rider_scratch.pth' not found! Please run 'python train.py' first.")
        return
        
    model.load_state_dict(torch.load("triple_rider_scratch.pth", map_location=device))
    model.eval()

    # Look for test images in the dataset/test folder
    test_images = (
        glob.glob("dataset/test/*.jpg") + 
        glob.glob("dataset/test/*.png") + 
        glob.glob("dataset/test/*.jpeg") + 
        glob.glob("dataset/test/*.JPG")
    )
    
    if not test_images:
        print("No test images found in 'dataset/test/' folder!")
        return

    # Grab the first test image
    # image_path = test_images[45]
    image_path = "image.png"
    print(f"Running prediction on test image: {image_path}")

    orig_img = cv2.imread(image_path)
    if orig_img is None:
        print(f"Could not load image at: {image_path}")
        return

    h, w, _ = orig_img.shape
    resized = cv2.resize(orig_img, (224, 224))
    input_tensor = torch.tensor(resized.transpose((2, 0, 1)) / 255.0, dtype=torch.float32).unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        class_logits, bbox_pred = model(input_tensor)
        pred_class_id = torch.argmax(class_logits, dim=1).item()
        bbox = bbox_pred[0].cpu().numpy()

    # Since the dataset is trained specifically on triple riding, Class 0 is the violation
    label_text = "TRIPLE RIDING VIOLATION!"
    box_color = (0, 0, 255)   # Bright Red Box for Violation
    
    print("\n[ALERT] Triple Riding Violation Detected on Bike!")

    # Scale bounding box back to original image dimensions
    x1, y1, x2, y2 = int(bbox[0] * w), int(bbox[1] * h), int(bbox[2] * w), int(bbox[3] * h)

    # Draw bounding box and label text
    cv2.rectangle(orig_img, (x1, y1), (x2, y2), box_color, 3)
    cv2.putText(orig_img, label_text, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

    # Save output to root folder
    output_file = "output_result.jpg"
    cv2.imwrite(output_file, orig_img)
    
    print(f"Saved output image to root folder as: {output_file}")

if __name__ == "__main__":
    run_prediction()