# ==== Imports ====
import os
import glob
import math
import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter
import numpy as np

# ==== Config ====
print("Initializing configuration...")
ROOT_DIR = "/kaggle/input/frames"
CHECKPOINT_PATH = "checkpoint_transformer.pth"
FINAL_MODEL_PATH = "video_transformer_final.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 16
BATCH_SIZE = 4
MAX_FRAMES = 200
EMBED_DIM = 512
PATIENCE = 5
NUM_HEADS = 8
NUM_LAYERS = 2

print(f"Using device: {DEVICE}")
print(f"Loading dataset from: {ROOT_DIR}")

# ==== Transforms ====
img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ==== Label Processing ====
label_names = sorted(os.listdir(ROOT_DIR))
label_map = {name: idx for idx, name in enumerate(label_names)}
NUM_CLASSES = len(label_names)
print(f"Found {NUM_CLASSES} classes: {label_names}")

# ==== Dataset ====
class VideoFrameDataset(Dataset):
    def __init__(self, samples, transform=None, max_frames=MAX_FRAMES):
        self.samples = samples
        self.max_frames = max_frames
        print(f"VideoFrameDataset initialized with {len(samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label, transform = self.samples[idx]
        frame_paths = sorted(glob.glob(os.path.join(video_path, "*.jpg")))
        if len(frame_paths) == 0:
            for ext in ['*.png', '*.jpeg', '*.bmp']:
                frame_paths += sorted(glob.glob(os.path.join(video_path, ext)))
        if len(frame_paths) == 0:
            raise RuntimeError(f"No frames found in {video_path}")

        if len(frame_paths) > self.max_frames:
            indices = torch.randperm(len(frame_paths))[:self.max_frames]
            frame_paths = [frame_paths[i] for i in sorted(indices.tolist())]
        else:
            frame_paths += [frame_paths[-1]] * (self.max_frames - len(frame_paths))

        frames = []
        for p in frame_paths:
            try:
                img = Image.open(p).convert("RGB")
                frames.append(transform(img))
            except:
                raise RuntimeError(f"Error reading frame {p}")
        return torch.stack(frames), label

# ==== Model ====
print("Building model components...")
class FrameFeatureExtractor(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.project = nn.Linear(resnet.fc.in_features, embed_dim)

    def forward(self, x):
        with torch.no_grad():
            x = self.encoder(x.float())
        x = x.view(x.size(0), -1)
        return self.project(x)

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class VideoTransformerClassifier(nn.Module):
    def __init__(self, num_frames, embed_dim, num_classes, num_heads, num_layers):
        super().__init__()
        print(f"Initializing Transformer model: {num_layers} layers, {num_heads} heads")
        self.embed_dim = embed_dim
        self.frame_encoder = FrameFeatureExtractor(embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=num_frames)
        self.query_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.attn_pool = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        encoder_layer = nn.TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward=1024, dropout=0.2, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, num_classes))

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = self.frame_encoder(x)
        x = x.view(B, T, self.embed_dim)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        query = self.query_token.expand(B, -1, -1)
        attn_out, _ = self.attn_pool(query, x, x)
        return self.classifier(attn_out.squeeze(1))

# ==== Data Split ====
print("Splitting dataset into train/test sets...")
samples = [(os.path.join(ROOT_DIR, label, v), label_map[label])
           for label in label_names for v in os.listdir(os.path.join(ROOT_DIR, label))]
train_samples, test_samples = train_test_split(samples, test_size=0.2, stratify=[s[1] for s in samples], random_state=42)

AUGMENT_SETS = [
    transforms.Compose([transforms.Resize((224, 224)), transforms.RandomRotation(5), transforms.ToTensor()]),
    transforms.Compose([transforms.Resize((224, 224)), transforms.ColorJitter(brightness=0.2), transforms.ToTensor()]),
    transforms.Compose([transforms.Resize((224, 224)), transforms.GaussianBlur(3), transforms.ToTensor()]),
    transforms.Compose([transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(p=1.0), transforms.ToTensor()]),
    transforms.Compose([transforms.Resize((224, 224)), transforms.RandomVerticalFlip(p=1.0), transforms.ToTensor()]),
]

train_samples = []
for label in label_names:
    label_dir = os.path.join(ROOT_DIR, label)
    video_dirs = os.listdir(label_dir)
    
    for vid in video_dirs:
        orig_path = os.path.join(label_dir, vid)
        if not os.path.isdir(orig_path):
            continue

        train_samples.append((orig_path, label_map[label], img_transform))
        for aug in AUGMENT_SETS:
            train_samples.append((orig_path, label_map[label], aug))

print(f"Train samples: {len(train_samples)} | Test samples: {len(test_samples)}")

label_counts = Counter([s[1] for s in train_samples])
total = sum(label_counts.values())
weights = [total / label_counts[i] for i in range(NUM_CLASSES)]
class_weights = torch.FloatTensor(weights).to(DEVICE)

train_dataset = VideoFrameDataset(train_samples, max_frames=MAX_FRAMES)
test_dataset = VideoFrameDataset([(p, l, img_transform) for p, l in test_samples], max_frames=MAX_FRAMES)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# ==== Init ====
print("Initializing model, optimizer, and loss...")
model = VideoTransformerClassifier(MAX_FRAMES, EMBED_DIM, NUM_CLASSES, NUM_HEADS, NUM_LAYERS).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss(weight=class_weights)

start_epoch = 0
best_acc = 0
patience_counter = 0
if os.path.exists(CHECKPOINT_PATH):
    print("Loading checkpoint...")
    cp = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(cp["model_state_dict"])
    optimizer.load_state_dict(cp["optimizer_state_dict"])
    start_epoch = cp["epoch"] + 1
    best_acc = cp["best_acc"]
    print(f"Resumed from epoch {start_epoch}")

train_losses, train_accuracies, test_accuracies = [], [], []

# ==== Training ====
print("Starting training loop...")
for epoch in range(start_epoch, NUM_EPOCHS):
    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
    model.train()
    total_loss, correct, total = 0, 0, 0
    for videos, labels in tqdm(train_loader, desc=f"Train {epoch+1}"):
        videos, labels = videos.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        output = model(videos)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (output.argmax(1) == labels).sum().item()
        total += labels.size(0)
    train_acc = correct / total
    train_losses.append(total_loss / len(train_loader))
    train_accuracies.append(train_acc)

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for videos, labels in test_loader:
            videos, labels = videos.to(DEVICE), labels.to(DEVICE)
            preds = model(videos).argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    test_acc = correct / total
    test_accuracies.append(test_acc)

    print(f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc
    }, CHECKPOINT_PATH)

    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), FINAL_MODEL_PATH)
        torch.save(model, "model_full.pth")
        print("Best model saved.")
        patience_counter = 0
    else:
        patience_counter += 1
        print(f"No improvement. Patience counter: {patience_counter}/{PATIENCE}")
        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

# ==== Visualization ====
def plot_loss_accuracy(train_losses, train_accuracies, test_accuracies):
    print("Plotting loss and accuracy...")
    epochs = range(1, len(train_losses)+1)
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(epochs, train_losses, marker='o', label='Train Loss')
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.grid()
    plt.subplot(1,2,2)
    plt.plot(epochs, train_accuracies, marker='o', label='Train Acc')
    plt.plot(epochs, test_accuracies, marker='x', label='Test Acc')
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.grid()
    plt.legend()
    plt.show()

plot_loss_accuracy(train_losses, train_accuracies, test_accuracies)

# ==== Evaluation ====
def predict_and_evaluate(model, dataloader, device, class_names):
    print("Running evaluation on test set...")
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for videos, labels in dataloader:
            videos, labels = videos.to(device), labels.to(device)
            outputs = model(videos).argmax(1)
            y_pred.extend(outputs.cpu().numpy())
            y_true.extend(labels.cpu().numpy())
    print("Classification Report:\n")
    print(classification_report(y_true, y_pred, target_names=class_names))
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

predict_and_evaluate(model, test_loader, DEVICE, label_names)

# ==== Load Model and Predict ====
def load_model_and_predict(video_tensor):
    print("Loading model for inference...")
    model = VideoTransformerClassifier(MAX_FRAMES, EMBED_DIM, NUM_CLASSES, NUM_HEADS, NUM_LAYERS).to(DEVICE)
    model.load_state_dict(torch.load(FINAL_MODEL_PATH))
    model.eval()
    with torch.no_grad():
        pred = model(video_tensor.unsqueeze(0).to(DEVICE))
        return pred.argmax(1).item()
