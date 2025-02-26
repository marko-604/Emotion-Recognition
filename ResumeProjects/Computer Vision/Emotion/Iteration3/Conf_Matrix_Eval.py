import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")  # Use a non-interactive backend
import matplotlib.pyplot as plt



# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the same image transformations as training
img_size = 48
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Load the validation dataset
data_dir = "../train/"
dataset = datasets.ImageFolder(root=data_dir, transform=transform)
_, val_dataset = torch.utils.data.random_split(dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))])
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Define the model architecture
class EmotionCNN(torch.nn.Module):
    def __init__(self):
        super(EmotionCNN, self).__init__()
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Dropout(0.2),

            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Dropout(0.2),

            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Dropout(0.2),
        )
        self.fc_layers = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(256 * 6 * 6, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(512, 7)  # 7 emotion classes
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# Load the trained model
model = EmotionCNN().to(device)
model.load_state_dict(torch.load("../Iteration1/emotion_model.pth", map_location=device))
model.eval()

# Get predictions on the validation set
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Compute confusion matrix
cm = confusion_matrix(all_labels, all_preds)
classes = dataset.classes  # Emotion labels

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
print("Confusion matrix saved as 'confusion_matrix.png'.")
