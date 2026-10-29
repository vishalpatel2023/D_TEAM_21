import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np

# 1. Custom CNN Architecture Built Completely From Scratch
class TripleRiderScratchCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(TripleRiderScratchCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classification head (3 classes)
        self.classifier = nn.Linear(128, num_classes)
        
        # Bounding box regressor [x1, y1, x2, y2]
        self.bbox_regressor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        
        class_logits = self.classifier(x)
        bbox_coords = self.bbox_regressor(x)
        return class_logits, bbox_coords

# 2. Dataset loader for Darknet (Images and .txt files in the same folder)
class DarknetDataset(Dataset):
    def __init__(self, data_dir):
        # Find all .jpg and .png images in this folder
        self.img_paths = sorted(
            glob.glob(os.path.join(data_dir, "*.jpg")) + 
            glob.glob(os.path.join(data_dir, "*.png")) +
            glob.glob(os.path.join(data_dir, "*.jpeg"))
        )
        self.data_dir = data_dir

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        
        # The .txt file has the exact same name as the image, just a different extension
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(self.data_dir, base_name + ".txt")

        # Load image
        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        img = cv2.resize(img, (224, 224))
        img = img.transpose((2, 0, 1)) / 255.0  # Normalize to [0, 1]
        img_tensor = torch.tensor(img, dtype=torch.float32)

        # Default label values
        class_id = 0
        bbox = [0.1, 0.1, 0.9, 0.9]

        # Read the .txt file if it exists
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                line = f.readline().strip().split()
                if len(line) >= 5:
                    class_id = int(line[0])
                    xc, yc, w, h = map(float, line[1:5])
                    bbox = [max(0.0, xc - w/2), max(0.0, yc - h/2), min(1.0, xc + w/2), min(1.0, yc + h/2)]

        return img_tensor, torch.tensor(class_id, dtype=torch.long), torch.tensor(bbox, dtype=torch.float32)

# 3. Training Loop
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Point directly to the 'train' folder where images and .txt files live together
    dataset = DarknetDataset("dataset/train")
    if len(dataset) == 0:
        print("Error: No images found in dataset/train/! Please check your folder path.")
        return

    print(f"Found {len(dataset)} training images.")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    model = TripleRiderScratchCNN(num_classes=3).to(device)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_bbox = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Starting training from scratch...")
    model.train()
    epochs = 10

    for epoch in range(epochs):
        total_loss = 0.0
        for images, labels_cls, labels_bbox in dataloader:
            images, labels_cls, labels_bbox = images.to(device), labels_cls.to(device), labels_bbox.to(device)

            optimizer.zero_grad()
            pred_cls, pred_bbox = model(images)

            loss = criterion_cls(pred_cls, labels_cls) + criterion_bbox(pred_bbox, labels_bbox)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {total_loss / len(dataloader):.4f}")

    torch.save(model.state_dict(), "triple_rider_scratch.pth")
    print("Training complete! Model weights saved to 'triple_rider_scratch.pth'.")

if __name__ == "__main__":
    train_model()