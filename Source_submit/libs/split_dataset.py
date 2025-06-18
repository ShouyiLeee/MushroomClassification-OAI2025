"""
Script to split training data into train and validation sets using k-fold cross-validation
"""

import os
import random
import shutil
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold


def set_all_seeds(seed=42):
    """
    Set all seeds to ensure reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def create_k_fold_splits(train_path, val_path, n_splits=5, seed=42):
    """
    Create k-fold cross-validation splits of the dataset

    Args:
        train_path (str): Path to training data directory
        val_path (str): Path to validation data directory
        n_splits (int): Number of folds (default: 5)
        seed (int): Random seed for reproducibility (default: 42)
    """
    # Set all seeds for reproducibility
    set_all_seeds(seed)

    # Get all class directories
    class_dirs = sorted([d for d in os.listdir(train_path)
                         if os.path.isdir(os.path.join(train_path, d))])

    print(f"Found {len(class_dirs)} classes in training data")

    # First pass: collect all images and their classes
    class_images = {}
    total_images = 0

    for class_dir in class_dirs:
        train_class_dir = os.path.join(train_path, class_dir)
        images = sorted([f for f in os.listdir(train_class_dir)
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        class_images[class_dir] = images
        total_images += len(images)

    print(f"\nTotal images across all classes: {total_images}")

    # Create k-fold splits for each class
    for fold in range(n_splits):
        print(f"\nCreating fold {fold + 1}/{n_splits}")

        # Create fold-specific validation directory
        fold_val_path = os.path.join(val_path, f'fold_{fold+1}')
        os.makedirs(fold_val_path, exist_ok=True)

        # Process each class
        for class_dir in class_dirs:
            print(f"\nProcessing class: {class_dir}")

            # Create class directory in validation set
            val_class_dir = os.path.join(fold_val_path, class_dir)
            os.makedirs(val_class_dir, exist_ok=True)

            # Get images for this class
            images = class_images[class_dir]

            # Create k-fold split for this class
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
            splits = list(kf.split(images))
            train_idx, val_idx = splits[fold]

            # Get validation images for this fold
            val_images = [images[i] for i in val_idx]

            print(f"Total images in class: {len(images)}")
            print(f"Moving {len(val_images)} images to validation set")

            # Copy selected images to validation directory
            for img in val_images:
                src = os.path.join(train_path, class_dir, img)
                dst = os.path.join(val_class_dir, img)
                # Use copy instead of move to keep original files
                shutil.copy2(src, dst)

            print(f"Copied {len(val_images)} images to validation set")

        # Print fold distribution
        print(f"\nFold {fold + 1} distribution:")
        print("Class\t\tTrain\tVal\tTotal")
        print("-" * 40)
        for class_dir in class_dirs:
            train_count = len(os.listdir(os.path.join(train_path, class_dir)))
            val_count = len(os.listdir(os.path.join(fold_val_path, class_dir)))
            total = train_count
            print(f"{class_dir}\t\t{train_count}\t{val_count}\t{total}")


def main():
    # Set all seeds at the start
    set_all_seeds()

    # Set paths
    dataset_path = os.path.join(os.getcwd(), 'Dataset')
    train_path = os.path.join(dataset_path, 'train')
    val_path = os.path.join(dataset_path, 'val')

    # Verify training directory exists
    if not os.path.exists(train_path):
        print(f"Error: Training directory not found at {train_path}")
        return

    # Create k-fold splits
    print("Starting k-fold dataset split...")
    create_k_fold_splits(train_path, val_path)
    print("\nK-fold dataset split completed!")


if __name__ == "__main__":
    main()
