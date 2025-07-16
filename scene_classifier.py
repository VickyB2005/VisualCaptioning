# 1. Extract frames from videos and organize by scene label (train/val split)
import os
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from PIL import Image

# --- PARAMETERS ---
video_dir = 'videos'  # folder with your videos
csv_file = 'labels.csv'  # CSV with columns: filename,scene
output_dir = 'frames_dataset'  # where to save extracted frames
num_epochs = 10
batch_size = 32
lr = 0.001

# --- EXTRACT FRAMES AND ORGANIZE ---
os.makedirs(output_dir, exist_ok=True)
df = pd.read_csv(csv_file)
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['scene'], random_state=42)

def extract_and_save(df, split):
    for idx, row in df.iterrows():
        video_path = os.path.join(video_dir, row['filename'])
        scene = row['scene']
        scene_dir = os.path.join(output_dir, split, scene)
        os.makedirs(scene_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count == 0:
            print(f"Warning: {video_path} has 0 frames.")
            continue
        mid_frame = frame_count // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
        ret, frame = cap.read()
        if ret:
            frame_filename = f"{os.path.splitext(row['filename'])[0]}_frame.jpg"
            cv2.imwrite(os.path.join(scene_dir, frame_filename), frame)
        else:
            print(f"Warning: Could not read frame from {video_path}")
        cap.release()

extract_and_save(train_df, 'train')
extract_and_save(val_df, 'val')
print("Frames extracted and organized into train/val splits.")

# --- TRAIN SCENE CLASSIFIER ---
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
data_dir = output_dir
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=2)
               for x in ['train', 'val']}
class_names = image_datasets['train'].classes
num_classes = len(class_names)
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, num_classes)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}')
    print('-' * 10)
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
        else:
            model.eval()
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        epoch_loss = running_loss / len(image_datasets[phase])
        epoch_acc = running_corrects.double() / len(image_datasets[phase])
        print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
    print()
torch.save(model.state_dict(), 'scene_classifier.pth')
print("Training complete. Model saved as scene_classifier.pth.")

# --- INFERENCE FUNCTION ---
def predict_scene(image_path, model, class_names):
    model.eval()
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = transform(img).unsqueeze(0).to(next(model.parameters()).device)
    with torch.no_grad():
        outputs = model(input_tensor)
        _, preds = torch.max(outputs, 1)
    return class_names[preds.item()]

