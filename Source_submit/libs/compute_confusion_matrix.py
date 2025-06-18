import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tabulate import tabulate


class Logger:
    def __init__(self, filename="log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding='utf-8')
        # Write timestamp
        self.log.write(f"\n{'='*50}\n")
        self.log.write(
            f"Evaluation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.log.write(f"{'='*50}\n")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def compute_confusion_matrix(ground_truth_file, predicted_file, log_file="log.txt"):
    """
    Compute and print confusion matrix between ground truth and predicted labels.

    Args:
        ground_truth_file (str): Path to the ground truth CSV file
        predicted_file (str): Path to the predicted CSV file
        log_file (str): Path to save the log file
    """
    # Set up logging
    sys.stdout = Logger(log_file)

    # Read CSV files
    print("\n" + "="*50)
    print("📊 Loading Data...")
    print("="*50)
    print(f"Ground Truth File: {ground_truth_file}")
    print(f"Predicted File: {predicted_file}")

    gt_df = pd.read_csv(ground_truth_file)
    pred_df = pd.read_csv(predicted_file)

    # Sort both dataframes by image name to ensure alignment
    gt_df = gt_df.sort_values('id')
    pred_df = pred_df.sort_values('id')

    # Verify that both files have the same images
    if not (gt_df['id'] == pred_df['id']).all():
        print("\n⚠️  Warning: Image lists in ground truth and predicted files do not match exactly")
        # Find common images
        common_images = set(gt_df['id']).intersection(set(pred_df['id']))
        gt_df = gt_df[gt_df['id'].isin(common_images)].sort_values('id')
        pred_df = pred_df[pred_df['id'].isin(common_images)].sort_values('id')

    # Get unique classes and sort them
    classes = sorted(set(gt_df['type'].unique()) |
                     set(pred_df['type'].unique()))

    # Find incorrect predictions
    incorrect_predictions = []
    for idx, (true_label, pred_label, img_id) in enumerate(zip(gt_df['type'], pred_df['type'], gt_df['id'])):
        if true_label != pred_label:
            incorrect_predictions.append([
                img_id,
                f"Class {true_label}",
                f"Class {pred_label}"
            ])

    # Compute confusion matrix
    cm = confusion_matrix(gt_df['type'], pred_df['type'], labels=classes)

    # Create a DataFrame for better visualization
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)

    # Print confusion matrix
    print("\n" + "="*50)
    print("📈 Confusion Matrix:")
    print("="*50)
    # Format confusion matrix with class labels
    headers = [f"Pred {c}" for c in classes]
    rows = [[f"True {c}"] + list(row) for c, row in zip(classes, cm)]
    print(tabulate(rows, headers=headers, tablefmt="grid"))

    # Compute metrics
    total_samples = np.sum(cm)
    correct_predictions = np.trace(cm)
    accuracy = correct_predictions / total_samples

    # Compute per-class metrics
    class_metrics = []
    for i, class_name in enumerate(classes):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        tn = total_samples - tp - fp - fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision +
                                         recall) if (precision + recall) > 0 else 0

        class_metrics.append([
            f"Class {class_name}",
            f"{precision:.4f}",
            f"{recall:.4f}",
            f"{f1:.4f}"
        ])

    # Print overall accuracy
    print("\n" + "="*50)
    print("🎯 Overall Results:")
    print("="*50)
    print(f"Total Samples: {total_samples}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Incorrect Predictions: {total_samples - correct_predictions}")
    print(f"Accuracy: {accuracy:.4f} ({correct_predictions}/{total_samples})")

    # Print class metrics
    print("\n" + "="*50)
    print("📊 Class-wise Metrics:")
    print("="*50)
    headers = ["Class", "Precision", "Recall", "F1-Score"]
    print(tabulate(class_metrics, headers=headers, tablefmt="grid"))

    # Print incorrect predictions
    if incorrect_predictions:
        print("\n" + "="*50)
        print("❌ Incorrect Predictions:")
        print("="*50)
        headers = ["Image ID", "True Label", "Predicted Label"]
        print(tabulate(incorrect_predictions, headers=headers, tablefmt="grid"))
        print(f"\nTotal incorrect predictions: {len(incorrect_predictions)}")
    else:
        print("\n" + "="*50)
        print("✅ All predictions are correct!")
        print("="*50)
    print("\n")

    # Reset stdout and close log file
    sys.stdout = sys.__stdout__
    print(f"Results have been saved to {log_file}")


if __name__ == "__main__":
    # Example usage
    ground_truth_file = "submission-99.csv"
    predicted_file = "submission.csv"

    # Create logs directory if it doesn't exist
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"evaluation_{timestamp}.log")

    compute_confusion_matrix(ground_truth_file, predicted_file, log_file)
