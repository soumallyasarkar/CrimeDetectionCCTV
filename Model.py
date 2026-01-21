# =========================
# IMPORTS
# =========================
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report
from tqdm import tqdm

# =========================
# GPU OPTIMIZATION
# =========================

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    # Enable TF32 for Ampere/Hopper GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Enable cuDNN autotuner
    torch.backends.cudnn.benchmark = True
else:
    print("No GPU detected, using CPU")

# =========================
# CONFIGURATION
# =========================
class Config:
    # Folder that contains the 'Violence' and 'NonViolence' directories
    ROOT_DATA = "Dataset/Real-life-violence/Real Life Violence Dataset"

    VIOLENCE_DIR = os.path.join(ROOT_DATA, "Violence")
    NON_VIOLENCE_DIR = os.path.join(ROOT_DATA, "NonViolence")

    OUTPUT_DIR = "models"

    INPUT_SHAPE = (30, 128, 128, 3)  # (frames, height, width, channels)
    BATCH_SIZE = 8                  # Safe for 128x128 on MIG
    EPOCHS = 50
    BASE_LR = 1e-4

VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.m4v', '.wmv')

# =========================
# VIDEO FRAME LOADER
# =========================
def load_video_frames(video_path, target_size=(128, 128), max_frames=30):
    """
    Reads a video and extracts fixed number of frames.
    Pads frames if video is shorter.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        frames = []

        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, target_size)
            frame = frame.astype("float32") / 255.0
            frames.append(frame)

        cap.release()

        if len(frames) == 0:
            return None

        # Pad last frame if needed
        while len(frames) < max_frames:
            frames.append(frames[-1])

        return frames[:max_frames]
    except Exception as e:
        print(f"Error loading {video_path}: {e}")
        return None

# =========================
# MOTION COMPUTATION
# =========================
def compute_motion(frames):
    """
    Computes frame-to-frame difference to highlight motion
    """
    motion_frames = []
    for i in range(1, len(frames)):
        diff = np.abs(frames[i] - frames[i-1])
        motion_frames.append(diff)

    motion_frames.insert(0, motion_frames[0])
    return np.array(motion_frames)

# =========================
# LAZY LOADING DATASET
# =========================
class LazyVideoDataset(Dataset):
    """
    Dataset that loads videos on-the-fly instead of loading all into memory
    """
    def __init__(self, video_paths, labels):
        self.video_paths = video_paths
        self.labels = labels
    
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        # Load video on demand
        frames = load_video_frames(self.video_paths[idx])
        
        if frames is None:
            # Return zeros if video fails to load
            motion = np.zeros((30, 128, 128, 3), dtype=np.float32)
        else:
            motion = compute_motion(frames)
        
        # Convert to PyTorch format: (C, T, H, W)
        motion_tensor = torch.FloatTensor(motion).permute(3, 0, 1, 2)
        label_tensor = torch.FloatTensor([self.labels[idx]])
        
        return motion_tensor, label_tensor

# =========================
# DATASET PREPARATION
# =========================
def prepare_dataset():
    """
    Prepares file paths and labels without loading videos
    """
    def load_files(folder):
        return sorted([
            os.path.join(folder, f) for f in os.listdir(folder)
            if f.lower().endswith(VIDEO_EXTENSIONS)
        ])

    vio_files = load_files(Config.VIOLENCE_DIR)
    non_files = load_files(Config.NON_VIOLENCE_DIR)

    print(f"Found {len(vio_files)} violent videos")
    print(f"Found {len(non_files)} non-violent videos")

    # Simple split (900 train / 100 test)
    train_vio, test_vio = vio_files[:900], vio_files[900:1000]
    train_non, test_non = non_files[:900], non_files[900:1000]

    # Prepare paths and labels
    train_paths = train_vio + train_non
    train_labels = [1] * len(train_vio) + [0] * len(train_non)
    
    test_paths = test_vio + test_non
    test_labels = [1] * len(test_vio) + [0] * len(test_non)

    print(f"Training samples: {len(train_paths)}")
    print(f"Test samples: {len(test_paths)}")

    return train_paths, train_labels, test_paths, test_labels

# =========================
# 3D CNN MODEL
# =========================
class CNN3D(nn.Module):
    """
    Deep 3D CNN optimized for spatiotemporal learning
    """
    def __init__(self, input_channels=3):
        super(CNN3D, self).__init__()
        
        # -------- Block 1 --------
        self.block1 = nn.Sequential(
            nn.Conv3d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2))
        )
        
        # -------- Block 2 --------
        self.block2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2))
        )
        
        # -------- Block 3 --------
        self.block3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2))
        )
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Fully Connected Layers
        self.fc = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# =========================
# TRAINING FUNCTION
# =========================
def train_model(model, train_loader, criterion, optimizer, scheduler, epochs):
    """
    Training loop with proper progress tracking
    """
    model.train()
    
    for epoch in range(epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"{'='*50}")
        
        epoch_loss = 0
        correct = 0
        total = 0
        
        # Progress bar for batches
        pbar = tqdm(train_loader, desc=f"Training", unit="batch")
        
        for batch_idx, (inputs, labels) in enumerate(pbar):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            epoch_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
        
        avg_loss = epoch_loss / len(train_loader)
        accuracy = 100 * correct / total
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Accuracy: {accuracy:.2f}%")
        
        # Step the scheduler
        scheduler.step(avg_loss)
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(Config.OUTPUT_DIR, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  Checkpoint saved: {checkpoint_path}")

# =========================
# EVALUATION FUNCTION
# =========================
def evaluate_model(model, test_loader):
    """
    Evaluates model on test set
    """
    model.eval()
    
    y_true = []
    y_pred = []
    
    print("\nEvaluating on test set...")
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing", unit="batch"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            predicted = (outputs > 0.5).float()
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    # Classification report
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    print("\n" + "="*50)
    print("CLASSIFICATION REPORT")
    print("="*50)
    print(classification_report(y_true, y_pred, target_names=['Non-Violent', 'Violent']))

# =========================
# MAIN EXECUTION
# =========================
def main():
    # Create output directory
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    # Prepare dataset (without loading videos)
    train_paths, train_labels, test_paths, test_labels = prepare_dataset()
    
    # Create lazy-loading datasets
    train_dataset = LazyVideoDataset(train_paths, train_labels)
    test_dataset = LazyVideoDataset(test_paths, test_labels)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    
    # Initialize model
    print("\nInitializing model...")
    model = CNN3D(input_channels=3).to(device)
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.BASE_LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )
    
    # Train model
    print("\nStarting training...")
    train_model(model, train_loader, criterion, optimizer, scheduler, Config.EPOCHS)
    
    # Evaluate model
    evaluate_model(model, test_loader)
    
    # Save final model
    final_model_path = os.path.join(Config.OUTPUT_DIR, "violence_detection_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"\n{'='*50}")
    print(f"Final model saved to: {final_model_path}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
