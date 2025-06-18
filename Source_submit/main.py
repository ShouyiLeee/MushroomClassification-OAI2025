### START: CÁC KHAI BÁO CHÍNH - KHÔNG THAY ĐỔI ###
SEED = 0  # Số seed (Ban tổ chức sẽ công bố & thay đổi vào lúc chấm)
# Đường dẫn đến thư mục train
# (đúng theo cấu trúc gồm 4 thư mục cho 4 classes của ban tổ chức)
TRAIN_DATA_DIR_PATH = "data/train" #'data/train'
# Đường dẫn đến thư mục test
TEST_DATA_DIR_PATH = "data/test" #'data/test'
### END: CÁC KHAI BÁO CHÍNH - KHÔNG THAY ĐỔI ###

import glob

### START: CÁC THƯ VIỆN IMPORT ###
# Lưu ý: các thư viện & phiên bản cài đặt vui lòng để trong requirements.txt
import os
import random
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import timm
import torch  # !!!
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, models, transforms
from tqdm.notebook import tqdm  # For progress bars in notebooks

### END: CÁC THƯ VIỆN IMPORT ###

### START: SEEDING EVERYTHING - KHÔNG THAY ĐỔI ###
# Seeding nhằm đảm bảo kết quả sẽ cố định
# và không ngẫu nhiên ở các lần chạy khác nhau
# Set seed for random
random.seed(SEED)
# Set seed for numpy
np.random.seed(SEED)
# Set seed for torch
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
# Set seed for tensorflow
tf.random.set_seed(SEED)
### END: SEEDING EVERYTHING - KHÔNG THAY ĐỔI ###



# START: IMPORT CÁC THƯ VIỆN CUSTOM, MODEL, v.v. riêng của nhóm ###
import libs.edit_enter as edit_enter
import libs.enter
import libs.train_model_vit as train_model_vit
from libs.multi_image_dataset import MultiImageMushroomDataset
from libs.multi_vit_classifier import ViT_MushroomClassifier

### END: IMPORT CÁC THƯ VIỆN CUSTOM, MODEL, v.v. riêng của nhóm ###


### START: ĐỊNH NGHĨA & CHẠY HUẤN LUYỆN MÔ HÌNH ###
# Model sẽ được train bằng cac ảnh ở [TRAIN_DATA_DIR_PATH]
#------------------------------------------------------------------------------------------------------------------------------
def copy_folder(source_path, destination_path, overwrite=True):
    """
    Copy a folder and its contents to a new location.

    Args:
        source_path (str): Path to the source folder
        destination_path (str): Path to the destination folder
        overwrite (bool): If True, overwrite existing files. If False, skip existing files.

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Convert paths to Path objects
        source = Path(source_path)
        destination = Path(destination_path)

        # Check if source exists
        if not source.exists():
            print(f"Error: Source path '{source_path}' does not exist")
            return False

        # Create destination directory if it doesn't exist
        destination.mkdir(parents=True, exist_ok=True)

        # Copy the folder
        if overwrite:
            # If overwrite is True, remove existing destination folder first
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(source, destination)
        else:
            # If overwrite is False, copy only if destination doesn't exist
            if not destination.exists():
                shutil.copytree(source, destination)
            else:

                print(
                    f"Destination folder '{destination_path}' already exists. Skipping...")

        print(f"Successfully copied '{source_path}' to '{destination_path}'")
        return True

    except Exception as e:
        print(f"Error copying folder: {str(e)}")
        return False


path_to_train_data = TRAIN_DATA_DIR_PATH  # ! Train path
source_path = path_to_train_data
destination_path = "Dataset/train"
copy_folder(source_path, destination_path, overwrite=True)

path_to_test_data = TEST_DATA_DIR_PATH  # ! Test path
source_path = path_to_test_data
destination_path = "Dataset/test"
copy_folder(source_path, destination_path, overwrite=True)


# Running the split_dataset.py script located in the libs folder
subprocess.run(['python', 'libs/split_dataset.py'])


for i in range(1):  # ! Run x time
    # val_fold, filter_train, change val is train
    edit_enter.create_enter_file(i+1, False, True)
    subprocess.run(['python', 'libs/train_model_cnn.py'])  # ! Script

    subprocess.run(['python', 'libs/train_model.py'])  # ! Script


subprocess.run(['python', 'libs/join_valid_test.py'])
subprocess.run(['python', 'libs/train_join.py'])

#---------------------------------------------------------------------------------------------------------------
# Truong's code
num_classes = 4
batch_size = 8
num_epochs = 20
learning_rate = 5e-6

# loss và optimizer
criterion = nn.CrossEntropyLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


transform_original = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

transform_augmented = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(degrees=10),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ColorJitter(brightness=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])



#-------------------------------------------------------------------------------------------------------------------------------------
def dataset_prepare(train_dir, transform, num_images_per_sample):
    dataset = MultiImageMushroomDataset(
        root_dir=train_dir,
        transform=transform,
        num_images_per_sample=num_images_per_sample
    )

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 to avoid pickle errors
        pin_memory=True,    # Faster data transfer to GPU
        drop_last=False,          # Use all samples
        persistent_workers=False,  # Keep workers alive between epochs
    )

    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle validation data
        num_workers=0,
        pin_memory=True,
    )
    return train_loader, val_loader
 #-------------------------------------------------------------------------------------------------------------------------------------

train_loader_2, val_loader_2 = dataset_prepare(
    train_dir=TRAIN_DATA_DIR_PATH,
    transform=transform_augmented,
    num_images_per_sample=2)


model_2 = ViT_MushroomClassifier(vit_model_name='vit_base_patch16_224', num_classes=num_classes).to(device)
optimizer_2 = torch.optim.AdamW(model_2.parameters(), lr=learning_rate)
model_2 = train_model_vit.training(model_2, optimizer_2, criterion, train_loader_2, val_loader_2, num_epochs=num_epochs, device=str(device), min_loss_threshold=0.008)
train_model_vit.evaluate(model_2, val_loader_2)
torch.save(model_2.state_dict(), "vit_mushroom_multi_2_best.pth")
#-------------------------------------------------------------------------------------

train_loader_3, val_loader_3 = dataset_prepare(
    train_dir=TRAIN_DATA_DIR_PATH,
    transform=transform_augmented,
    num_images_per_sample=3)

model_3 = ViT_MushroomClassifier(vit_model_name='vit_base_patch16_224', num_classes=num_classes).to(device)
optimizer_3 = torch.optim.AdamW(model_3.parameters(), lr=learning_rate)
model_3 = train_model_vit.training(model_3, optimizer_3, criterion, train_loader_3, val_loader_3, num_epochs=num_epochs, device=str(device), min_loss_threshold=0.001)
train_model_vit.evaluate(model_3, val_loader_3)
torch.save(model_3.state_dict(), "vit_mushroom_multi_3_best.pth")
#-------------------------------------------------------------------------------------

train_loader_4, val_loader_4 = dataset_prepare(
    train_dir=TRAIN_DATA_DIR_PATH,
    transform=transform_augmented,
    num_images_per_sample=4)

model_4 = ViT_MushroomClassifier(vit_model_name='vit_base_patch16_224', num_classes=num_classes).to(device)
optimizer_4 = torch.optim.AdamW(model_4.parameters(), lr=learning_rate)
model_4 = train_model_vit.training(model_4, optimizer_4, criterion, train_loader_4, val_loader_4, num_epochs=num_epochs, device=str(device), min_loss_threshold=0.001)
train_model_vit.evaluate(model_4, val_loader_4)
torch.save(model_4.state_dict(), "vit_mushroom_multi_4_best.pth")
#-------------------------------------------------------------------------------------

train_loader_5, val_loader_5 = dataset_prepare(
    train_dir=TRAIN_DATA_DIR_PATH,
    transform=transform_augmented,
    num_images_per_sample=5)


model_5 = ViT_MushroomClassifier(vit_model_name='vit_base_patch16_224', num_classes=num_classes).to(device)
optimizer_5 = torch.optim.AdamW(model_5.parameters(), lr=learning_rate)
model_5 = train_model_vit.training(model_5, optimizer_5, criterion, train_loader_5, val_loader_5, num_epochs=num_epochs, device=str(device), min_loss_threshold=0.001)
train_model_vit.evaluate(model_5, val_loader_5)
torch.save(model_5.state_dict(), "vit_mushroom_multi_5_best.pth")
 #-------------------------------------------------------------------------------------


### END: ĐỊNH NGHĨA & CHẠY HUẤN LUYỆN MÔ HÌNH ###

### START: THỰC NGHIỆM & XUẤT FILE KẾT QUẢ RA CSV ###
# Kết quả dự đoán của mô hình cho tập dữ liệu các ảnh ở [TEST_DATA_DIR_PATH]
# sẽ lưu vào file "output/results.csv"
# Cấu trúc gồm 2 cột: image_name và label: (encoded: 0, 1, 2, 3)
# image_name,label
# image1.jpg,0
# image2.jpg,1
# image3.jpg,2
# image4.jpg,3
#-----------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------
def extract_num_images_from_model_name(model):
    """
    Hàm phụ để trích xuất num_images_per_sample từ model.name (ví dụ: "model_3" → 3).
    """
    match = re.search(r'(\d+)', getattr(model, "name", "1"))
    return int(match.group(1)) if match else 1

def predict_single_image(model, image_path, transform, num_images_per_sample, device):
    """
    Fixed function to handle single image prediction with a multi-image model.
    """
    # Load and transform the image
    image = Image.open(image_path).convert("RGB")
    image = transform(image)

    # Create a batch with 2 copies of the same image to match expected shape [B, N, C, H, W]
    # where N is num_images_per_sample (2 in your case)
    image = torch.stack([image] * num_images_per_sample).unsqueeze(0).to(device)  # Shape: [1, N, C, H, W]

    model.eval()
    with torch.no_grad():
        output = model(image)
        pred = torch.argmax(output, dim=1).item()

    # Access class names from the dataset
    # class_names = ["Mỡ", "Bào Ngư", "Đùi Gà", "Linh Chi Trắng"]
    # return class_names[pred]
    return pred

def ensemble_predict_single_image(models, image_path, transform, device, method="soft"):
    """
    Dự đoán nhãn của một ảnh bằng cách ensemble nhiều mô hình, tự động lấy num_images_per_sample từ tên model.
    """
    predictions = []
    prob_sum = None

    for model in models:
        model.eval()
        num_images_per_sample = extract_num_images_from_model_name(model)

        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image)
        image_tensor = torch.stack([image_tensor] * num_images_per_sample).unsqueeze(0).to(device)  # [1, N, C, H, W]

        with torch.no_grad():
            output = model(image_tensor)
            probs = F.softmax(output, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            predictions.append(pred)

            if prob_sum is None:
                prob_sum = probs
            else:
                prob_sum += probs

    if method == "majority":
        raise NotImplementedError("Không hỗ trợ trả về prob cho phương pháp majority voting.")
    elif method == "soft":
        if prob_sum is None:
            raise ValueError("No valid probability sums were calculated. Check model outputs.")
        avg_probs = prob_sum / len(models)   ##Extract to ensemble with Xception
        return avg_probs.squeeze(0).cpu().numpy() #Return Avg probs thay vì class
    else:
        raise ValueError("Unknown ensemble method. Use 'soft' or 'majority'.")



def ensemble_predict_folder_to_csv(models, test_dir, transform, device, method="soft", output_csv="submission_ensemble.csv"):
    results = []

    for filename in sorted(os.listdir(test_dir)):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            path = os.path.join(test_dir, filename)
            probs = ensemble_predict_single_image(
                models=models,
                image_path=path,
                transform=transform,
                device=device,
                method=method
            )  # probs: numpy array [num_classes]

            # file_id = os.path.splitext(filename)[0]
            result = {'image_name': filename}

            # Ghi xác suất từng lớp vào cột prob_0, prob_1, ...
            for i, p in enumerate(probs):
                result[f'prob_{i}'] = float(p)  # Làm tròn nhẹ cho dễ đọc

            results.append(result)

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Ensemble predictions with probabilities saved to {output_csv}")

# model_2 = ViT_MushroomClassifier(vit_model_name='vit_base_patch16_224', num_classes=num_classes).to(device)
# model_3 = ViT_MushroomClassifier(vit_model_name='vit_base_patch16_224', num_classes=num_classes).to(device)
# model_4 = ViT_MushroomClassifier(vit_model_name='vit_base_patch16_224', num_classes=num_classes).to(device)
# model_5 = ViT_MushroomClassifier(vit_model_name='vit_base_patch16_224', num_classes=num_classes).to(device)

model_2.load_state_dict(torch.load("vit_mushroom_multi_2_best.pth"))
model_3.load_state_dict(torch.load("vit_mushroom_multi_3_best.pth"))
model_4.load_state_dict(torch.load("vit_mushroom_multi_4_best.pth"))
model_5.load_state_dict(torch.load("vit_mushroom_multi_5_best.pth"))
models = [model_2, model_3, model_4, model_5]  # các mô hình đã được load lên device

MultiVIT_output_file = os.path.join(os.getcwd(), libs.enter.FINAL_TO_USE_DIR, "result_MultiVIT.csv")

ensemble_predict_folder_to_csv(
    models=models,
    test_dir=TEST_DATA_DIR_PATH, #Replace with your test directory path
    transform=transform_original,
    device=device,
    method="soft",  #"soft" hoặc "majority"
    output_csv=MultiVIT_output_file #Replace with your desired output file path
)


# Đọc 2 file CSV
result_XceptionVIT_df = pd.read_csv(os.path.join(os.getcwd(), libs.enter.FINAL_TO_USE_DIR, "result_XceptionVIT.csv"))
result_MultiVIT_df = pd.read_csv(os.path.join(os.getcwd(), libs.enter.FINAL_TO_USE_DIR, "result_MultiVIT.csv"))

# Đảm bảo tên cột giống nhau và sắp xếp
result_XceptionVIT_df = result_XceptionVIT_df.sort_values("image_name").reset_index(drop=True)
result_MultiVIT_df = result_MultiVIT_df.sort_values("image_name").reset_index(drop=True)

# Tính trung bình các xác suất
avg_df = result_XceptionVIT_df.copy()
for col in ['prob_0', 'prob_1', 'prob_2', 'prob_3']:
    avg_df[col] = (result_XceptionVIT_df[col] + result_MultiVIT_df[col]) / 2

# Tìm label có xác suất cao nhất
avg_df['label'] = avg_df[['prob_0', 'prob_1', 'prob_2', 'prob_3']].idxmax(axis=1)
avg_df['label'] = avg_df['label'].str.extract(r'(\d+)').astype(int)  # Chuyển 'prob_1' → 1

# Chỉ giữ lại 2 cột: image_name và label
final_df = avg_df[['image_name', 'label']]

# Xuất ra file CSV
final_df.to_csv("output/result.csv", index=False)
print("Done! File saved to output/results.csv")



### END: THỰC NGHIỆM & XUẤT FILE KẾT QUẢ RA CSV ###
