"""
Image Classification Model Training Script
This script trains a Vision Transformer (ViT) model for image classification using PyTorch.
"""

# =============================================================================
# Imports
# =============================================================================


import hashlib  # Add this import at the top of your file
import os
import random
import warnings
from collections import defaultdict
from datetime import datetime

import cv2
import enter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm  # <-- Import timm
import torch
import torch.nn as nn
import torch.nn.functional as F  # <-- Added F for F.relu
import torch.optim as optim
from PIL import Image, ImageFilter
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from transformers import ViTForImageClassification, ViTImageProcessor

# Suppress CUDA warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
warnings.filterwarnings('ignore', category=UserWarning)


# Initialize CUDA
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


# =============================================================================
# Configuration
# =============================================================================


# Paths
PROJECT_PATH = os.getcwd()
DATASET_PATH = os.path.join(PROJECT_PATH, 'Dataset')
train_path = os.path.join(DATASET_PATH, 'train')

# Import validation settings from enter.py
try:
    VAL_SELECTION = enter.VAL_SELECTION
    FILTER_VAL_IMAGES = enter.FILTER_VAL_IMAGES
    USE_VAL_IS_TRAIN = enter.USE_VAL_IS_TRAIN
except ImportError:
    print("Warning: enter.py not found, using default validation settings")
    VAL_SELECTION = "val/fold_1"  # Default validation fold
    FILTER_VAL_IMAGES = True      # Default to filtering validation images
    USE_VAL_IS_TRAIN = False

val_path = os.path.join(DATASET_PATH, VAL_SELECTION)  # Set val fold dir
test_path = os.path.join(DATASET_PATH, 'test')  # Set test dir
if USE_VAL_IS_TRAIN:
    val_path = train_path
    print("Changed val_path as train_path.")

# Print validation settings
print(f"\n[+] Validation Settings:")
print(f"Validation Selection: {VAL_SELECTION}")
print(f"Filter Validation Images: {FILTER_VAL_IMAGES}")
print(f"Final val path: {val_path}")
# print(f"Test Direction: {test_path}")

# Model parameters
IMG_SIZE = 224  # ! Tensor input size
BATCH_SIZE = 16  # Reduced batch size for ViT
EPOCHS = 10  # ! Changed to 10 epochs
LEARNING_RATE = 2e-5  # Slightly increased learning rate for faster convergence
WEIGHT_DECAY = 1e-3
NUM_WORKERS = 0
DROPOUT_RATE = 0.2

# Early stopping parameters
PATIENCE = 3     # Reduced patience for shorter training
MIN_DELTA = 1e-4  # Minimum change to qualify as an improvement

# ViT specific parameters
MODEL_NAME = "google/vit-base-patch16-224"  # Pre-trained ViT model

# =============================================================================
# Utility Functions
# =============================================================================


def set_all_seeds(seed=42):
    """Set all seeds to ensure reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    # Set seeds for transforms
    torch.manual_seed(seed)
    # Set seeds for data loading

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    return seed_worker


def verify_directory(path, name):
    """Verify if directory exists and contains files"""
    if not os.path.exists(path):
        print(f"Error: {name} directory not found at {path}")
        return False

    files = sorted(os.listdir(path))  # Sort files for consistency
    if not files:
        print(f"Error: {name} directory is empty at {path}")
        return False

    print(f"\n{name} directory contents:")
    cnt = 0
    for file in files:
        print(f"- {file}")
        cnt += 1
        if cnt > 10:
            break
    print(f"Total {len(files)} files in {path} directory")
    return True


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# =============================================================================
# Dataset Classes
# =============================================================================


def apply_enhance_image(image):
    # # Resize the image to (256, 256) and RGB
    # image = image.resize((256, 256))

    # # Apply Gaussian Blur
    # image = image.filter(ImageFilter.GaussianBlur(radius=1.5))

    # # Convert to numpy array for CLAHE
    # image_np = np.array(image)

    # # Apply CLAHE in RGB
    # # Split the image into R, G, B channels
    # r, g, b = cv2.split(image_np)

    # # Create CLAHE object
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # # Apply CLAHE to each channel
    # r_clahe = clahe.apply(r)
    # g_clahe = clahe.apply(g)
    # b_clahe = clahe.apply(b)

    # # Merge the channels back
    # image_np = cv2.merge((r_clahe, g_clahe, b_clahe))

    # # Convert back to PIL Image
    # image = Image.fromarray(image_np)
    return image


class TestDataset(Dataset):
    """Custom dataset for test images"""

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir)
                            if f.endswith(('.jpg', '.jpeg', '.png'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')  # Pil image

        # image = apply_enhance_image(image=image)  # ! Only Apply for test
        if self.transform:
            image = self.transform(image)
        return image, img_name

# =============================================================================
# Data Preparation
# =============================================================================


def create_data_loaders(train_path, val_path, test_path, img_size, batch_size, image_processor):
    """Create and configure data loaders with augmentation"""
    # Set worker seed function
    seed_worker = set_all_seeds()
    g = torch.Generator()
    g.manual_seed(42)

    # Training data augmentation - adjusted for small dataset
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ColorJitter(
            brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.RandomGrayscale(p=0.05),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
        transforms.ToTensor(),
    ])

    # Validation/Test data transformation
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    # Get validation image filenames to exclude from training
    val_images = set()
    if FILTER_VAL_IMAGES == True:
        if os.path.exists(val_path):
            for class_dir in sorted(os.listdir(val_path)):  # Sort for consistency
                val_class_path = os.path.join(val_path, class_dir)
                if os.path.isdir(val_class_path):
                    # Sort for consistency
                    val_images.update(sorted(os.listdir(val_class_path)))

    # Custom ImageFolder that excludes validation images
    class FilteredImageFolder(ImageFolder):
        def __init__(self, root, transform=None, val_images=None):
            self.val_images = val_images or set()
            super().__init__(root=root, transform=transform)
            # Filter out validation images from samples
            self.samples = [(path, label) for path, label in self.samples
                            if os.path.basename(path) not in self.val_images]
            self.imgs = self.samples

        def __getitem__(self, index):
            # Get the original item
            img_path, target = self.samples[index]
            return super().__getitem__(index=index), img_path
            # img = Image.open(img_path).convert('RGB')  # Load the image

            # # Apply image enhancement before any transformations
            # img = apply_enhance_image(img)
            # if self.transform:
            #     img = self.transform(img)
            # return img, target

    # Create datasets
    train_dataset = FilteredImageFolder(
        train_path, transform=train_transform, val_images=val_images)
    val_dataset = FilteredImageFolder(
        val_path, transform=val_transform, val_images=set())
    # val_dataset = ImageFolder(val_path, transform=val_transform)
    test_dataset = TestDataset(test_path, transform=val_transform)

    # Print dataset information
    print("\nDataset Information:")
    print(
        f"Total training images (after excluding validation): {len(train_dataset)}")
    print(f"Total validation images: {len(val_dataset)}")
    print(f"Total test images: {len(test_dataset)}")
    print(f"- Ordered classes in train dataset: {train_dataset.classes}")
    print(f"- Ordered classes in val dataset: {val_dataset.classes}")
    print("Class distribution in training set:")
    class_counts = defaultdict(int)
    for _, label in train_dataset.samples:
        class_counts[train_dataset.classes[label]] += 1
    for class_name, count in class_counts.items():
        print(f"- {class_name}: {count} images")

    # Create data loaders with worker_init_fn
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g
    )

    return train_loader, val_loader, test_loader, train_dataset.classes

# =============================================================================
# Model Creation
# =============================================================================


def create_model(num_classes):
    """Create and configure the Vision Transformer model"""
    # Load pre-trained ViT model
    model = ViTForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
        hidden_dropout_prob=DROPOUT_RATE,
        attention_probs_dropout_prob=DROPOUT_RATE
    )

    # Initialize image processor
    image_processor = ViTImageProcessor.from_pretrained(MODEL_NAME)

    # Unfreeze all layers for fine-tuning
    for param in model.parameters():
        param.requires_grad = True

    return model, image_processor

# =============================================================================
# Training Functions
# =============================================================================


def train_epoch(model, train_loader, criterion, optimizer, device, image_processor):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for (images, labels), img_paths in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Process images through ViT processor
        # Convert to PIL images first
        pil_images = [transforms.ToPILImage()(img) for img in images]
        inputs = image_processor(pil_images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        optimizer.zero_grad()
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / len(train_loader), 100. * correct / total


def validate(model, val_loader, criterion, device, image_processor, class_names, tmp_dir, timestamp):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    predictions = []  # List to store predictions and labels
    with torch.no_grad():
        for (images, labels), img_paths in val_loader:
            images, labels = images.to(device), labels.to(device)

            # Hash the images
            hash_ids = list(img_paths)

            # Process images through ViT processor
            # Convert to PIL images first
            pil_images = [transforms.ToPILImage()(img) for img in images]
            inputs = image_processor(pil_images, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs, labels=labels)
            loss = outputs.loss

            running_loss += loss.item()
            _, predicted = outputs.logits.max(1)
            probs = outputs.logits.cpu().numpy()
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Collect predictions and labels
            for hid, idy, prob in zip(hash_ids, labels.cpu().numpy(), probs):
                idx = np.argmax(prob)
                predictions.append(
                    [hid, class_names[idy], class_names[idx]] + prob.tolist())

    # Save predictions to CSV
    df = pd.DataFrame(predictions, columns=[
                      'image', 'actual', 'pred'] + class_names)
    submission_file = os.path.join(
        tmp_dir, f'validation_predictions_{timestamp}.csv')
    df.to_csv(submission_file, index=False)
    print(f"Validation predictions saved to: {submission_file}")

    # Calculate and print confusion matrix
    y_true = df['actual']
    y_pred = df['pred']
    cm = confusion_matrix(y_true, y_pred)
    # Print confusion matrix
    print("Confusion Matrix:")
    print(class_names)
    print(cm)

    return running_loss / len(val_loader), 100. * correct / total


def save_predictions(model, test_loader, class_names, device, output_dir, timestamp, image_processor):
    """Generate predictions for test images and save results"""
    model.eval()
    predictions = []

    with torch.no_grad():
        for images, filenames in test_loader:
            images = images.to(device)

            # Process images through ViT processor
            pil_images = [transforms.ToPILImage()(img) for img in images]
            inputs = image_processor(pil_images, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)
            _, predicted = outputs.logits.max(1)
            probs = outputs.logits.cpu().numpy()

            for filename, pred, prob in zip(filenames, predicted, probs):
                class_name = class_names[pred.item()]
                predictions.append([filename, class_name] + prob.tolist())

    # Save predictions to CSV
    df = pd.DataFrame(predictions, columns=['image', 'pred'] + class_names)
    submission_file = os.path.join(output_dir, f'test_{timestamp}.csv')
    df.to_csv(submission_file, index=False)
    print(f"\nPredictions saved to {submission_file}")


def plot_training_history(train_losses, val_losses, train_accs, val_accs, output_dir, timestamp):
    """Plot and save training history"""
    plt.figure(figsize=(12, 4))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(output_dir, f'training_history_{timestamp}.png')
    plt.savefig(plot_path)
    plt.close()

    print(f"\nTraining history plot saved to: {plot_path}")

# =============================================================================
# Main Execution
# =============================================================================


def main():
    # Set all seeds at the start
    set_all_seeds()

    # Print current directory and verify paths
    print("Current working directory:", os.getcwd())
    print("Project path:", PROJECT_PATH)
    print("Dataset path:", DATASET_PATH)

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(PROJECT_PATH, 'outputs', timestamp)  # Fixed
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory created at: {output_dir}")
    # Create tmp directory
    tmp_dir = os.path.join(PROJECT_PATH, 'tmps')  # Fixed
    os.makedirs(tmp_dir, exist_ok=True)
    print(f"Tmp directory created at: {tmp_dir}")

    # Verify directories
    if not verify_directory(train_path, "Train") or not verify_directory(val_path, "Validation") or not verify_directory(test_path, "Test"):
        print(
            "\nPlease ensure your dataset is properly organized in the following structure:")
        print("Dataset/")
        print("  ├── train/")
        print("  │   ├── class1/")
        print("  │   │   ├── image1.jpg")
        print("  │   │   └── ...")
        print("  │   ├── class2/")
        print("  │   └── ...")
        print("  ├── val/")
        print("  │   ├── class1/")
        print("  │   └── ...")
        print("  └── test/")
        print("      ├── 001.jpg")
        print("      └── ...")
        return

    # Check GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    print(f"PyTorch version: {torch.__version__}")

    # Create data loaders
    train_loader, val_loader, test_loader, class_names = create_data_loaders(
        train_path, val_path, test_path, IMG_SIZE, BATCH_SIZE, None
    )

    # Initialize model
    model, image_processor = create_model(len(class_names))
    model = model.to(device)

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(),
                            lr=LEARNING_RATE,
                            weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    # Training loop
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    print("\nStarting training for 10 epochs...")
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print("-" * 20)

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, image_processor)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, image_processor, class_names, tmp_dir, timestamp)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Early stopping with minimum improvement threshold
        if val_loss < best_val_loss - MIN_DELTA:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), os.path.join(
                output_dir, f'model_{timestamp}.pth'))
            print("✓ Saved new best model")
        else:
            patience_counter += 1

        # Print epoch results
        print(f"Training Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")
        print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")
        print(
            f"Best Validation Loss: {best_val_loss:.4f} (epoch {best_epoch+1})")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.2e}")

        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping triggered after epoch {epoch+1}")
            print(f"Best model was saved at epoch {best_epoch+1}")
            break

    # Print training summary
    print("\n[+] Training Summary:")
    print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch+1}")
    print(f"Best training accuracy: {train_accs[best_epoch]:.2f}%")
    print(f"Best validation accuracy: {val_accs[best_epoch]:.2f}%")

    # Save training history plot
    plot_training_history(train_losses, val_losses,
                          train_accs, val_accs, output_dir, timestamp)

    # ! Load the best model for predictions
    best_model_path = os.path.join(output_dir, f'model_{timestamp}.pth')
    if os.path.exists(best_model_path):
        print(
            f"\n[+] Loading best model from epoch {best_epoch+1}-th for predictions...")
        model.load_state_dict(torch.load(best_model_path, weights_only=True))
    else:
        print("\nWarning: Best model file not found, using current model state")

    # Generate and save predictions
    print("\nGenerating predictions on test set...")
    save_predictions(model, test_loader, class_names, device,
                     output_dir, timestamp, image_processor)

    print("\nTraining completed!")
    print(f"All outputs saved to: {output_dir}")
    print(
        f"Best model was from epoch {best_epoch+1} with validation loss: {best_val_loss:.4f}")
    print("+" * 100 + "\n")


if __name__ == "__main__":
    main()
