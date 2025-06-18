import glob
import os
import re
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, models, transforms
from tqdm.notebook import tqdm  # For progress bars in notebooks

train_dir="D:/IT/GITHUB/Hutech-AI-Challenge/data/train"
test_dir="D:/IT/GITHUB/Hutech-AI-Challenge/data/test"


def evaluate_and_return_loss(model, val_loader, criterion, device='cuda'):
    model.eval()
    val_loss = 0.0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            total += 1

    avg_val_loss = val_loss / total
    return avg_val_loss

def training(model, optimizer, criterion, train_loader, val_loader,
             num_epochs=10, device='cuda', min_loss_threshold=0.01):

    best_val_loss = float('inf')
    best_model_state = None  # Lưu trạng thái tốt nhất

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = evaluate_and_return_loss(model, val_loader, criterion, device)

        print(f"[Epoch {epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Evaluate on validation set (inference only, không tính loss)
        print("Validation performance:")
        # evaluate(model, val_loader)

        # Cập nhật mô hình tốt nhất
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
            print(f"✅ New best model saved at Epoch {epoch+1} with Val Loss: {best_val_loss:.4f}")

        # Stop nếu val_loss quá thấp
        if (avg_val_loss < min_loss_threshold) and (avg_train_loss < min_loss_threshold):
            print(f"🛑 Early stopping: Val Loss {avg_val_loss:.4f} and Train Loss {avg_train_loss:.4f} < Threshold {min_loss_threshold}")
            break

    # Load lại best model trước khi trả về
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("🔄 Loaded best model from training.")

    return model  # Trả về mô hình tốt nhất





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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 4
batch_size = 8
num_epochs = 20
# learning_rate = 1e-5

# loss và optimizer
criterion = nn.CrossEntropyLoss()


# Hàm đánh giá
def evaluate(model, dataloader):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    report = classification_report(y_true, y_pred, target_names=["bào ngư xám + trắng", "Đùi gà Baby (cắt ngắn)", "nấm mỡ", "linh chi trắng"])
    print(report)



class MultiImageMushroomDataset(Dataset):
    def __init__(self, root_dir, transform=None, num_images_per_sample=4):
        """
        Dataset phân loại nấm sử dụng nhiều ảnh một mẫu.

        Args:
            root_dir (str): Thư mục gốc chứa các class folders.
            transform (callable, optional): Transform áp dụng lên ảnh.
            num_images_per_sample (int): Số ảnh muốn nhóm lại thành 1 sample (mặc định 4).
        """
        self.samples = []
        self.transform = transform
        self.num_images_per_sample = num_images_per_sample

        # Map từ prefix sang class label
        self.prefix2class = {
            'NM': 'nấm mỡ',
            'BN': 'bào ngư xám + trắng',
            'DG': 'Đùi gà Baby (cắt ngắn)',
            'LC': 'linh chi trắng'
        }
        self.class_names = list(self.prefix2class.values())
        self.class2idx = {name: idx for idx, name in enumerate(self.class_names)}

        # Duyệt từng class folder
        for class_name in self.class_names:
            class_path = os.path.join(root_dir, class_name)
            images = sorted(glob.glob(os.path.join(class_path, "*.jpg")))

            # Gom ảnh theo prefix (BM, AB, ...)
            prefix_groups = {}
            for img_path in images:
                filename = os.path.basename(img_path)
                match = re.match(r"([A-Z]{2})\d+", filename)
                if match:
                    prefix = match.group(1)
                    if prefix not in prefix_groups:
                        prefix_groups[prefix] = []
                    prefix_groups[prefix].append(img_path)

            # Tạo samples từ các nhóm ảnh
            for prefix, img_list in prefix_groups.items():
                img_list = sorted(img_list)
                label_idx = self.class2idx[self.prefix2class[prefix]]

                for i in range(0, len(img_list), self.num_images_per_sample):
                    selected = img_list[i:i+self.num_images_per_sample]
                    if len(selected) < self.num_images_per_sample:
                        selected = (selected + [selected[0]] * self.num_images_per_sample)[:self.num_images_per_sample]
                    self.samples.append((selected, label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_paths, label = self.samples[idx]
        imgs = []

        for path in img_paths:
            image = Image.open(path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            imgs.append(image)

        return torch.stack(imgs), torch.tensor(label)

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


class ViT_MushroomClassifier(nn.Module):
    def __init__(self, vit_model_name='vit_base_patch16_224', num_classes=4):
        super(ViT_MushroomClassifier, self).__init__()
        self.vit = timm.create_model(vit_model_name, pretrained=True)
        self.vit.head = nn.Identity()  # Bỏ classification head của ViT

        self.embedding_dim = self.vit.num_features  # Thường là 768

        self.classifier = nn.Sequential(
            nn.Linear(self.embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # x: [B, 4, C, H, W]
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)

        embeddings = self.vit(x)  # [B*4, D]
        embeddings = embeddings.view(B, N, -1)  # [B, 4, D]

        # Mean pooling over 4 embeddings
        pooled = embeddings.mean(dim=1)  # [B, D]
        out = self.classifier(pooled)
        return out


 #Replace with your train directory path

#-------------------------------------------------------------------------

# dataset_augmented_1 = MultiImageMushroomDataset(
#     root_dir=train_dir,
#     transform=transform_augmented,
#     num_images_per_sample=1
# )


# train_loader_1= DataLoader(
#     dataset_augmented_1,
#     batch_size=batch_size,
#     shuffle=True,
#     num_workers=0,  # Set to 0 to avoid pickle errors
#     pin_memory=True,    # Faster data transfer to GPU
#     drop_last=False,          # Use all samples
#     persistent_workers=0,  # Keep workers alive between epochs
# )

# val_loader_1 = DataLoader(
#     dataset_augmented_1,
#     batch_size=batch_size,
#     shuffle=False,  # No need to shuffle validation data
#     num_workers=0,
#     pin_memory=True,
# )
# optimizer_1 = torch.optim.AdamW(model.parameters(), lr=learning_rate)
# model_1 = ViT_MushroomClassifier(vit_model_name='vit_base_patch16_224', num_classes=num_classes).to(device)
# model_1 = training(model_1, optimizer, criterion, train_loader_1, val_loader_1, num_epochs=num_epochs, device='cuda')
# evaluate(model_1, val_loader)
# torch.save(model_1.state_dict(), "vit_mushroom_multi_1_best.pth")
# #-------------------------------------------------------------------------------------
dataset_augmented_2 = MultiImageMushroomDataset(
    root_dir=train_dir,
    transform=transform_augmented,
    num_images_per_sample=2
)


train_loader_2= DataLoader(
    dataset_augmented_2,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,  # Set to 0 to avoid pickle errors
    pin_memory=True,    # Faster data transfer to GPU
    drop_last=False,          # Use all samples
    persistent_workers=0,  # Keep workers alive between epochs
)

val_loader_2 = DataLoader(
    dataset_augmented_2,
    batch_size=batch_size,
    shuffle=False,  # No need to shuffle validation data
    num_workers=0,
    pin_memory=True,
)

model_2 = ViT_MushroomClassifier(vit_model_name='vit_base_patch16_224', num_classes=num_classes).to(device)
optimizer_2 = torch.optim.AdamW(model_2.parameters(), lr=2e-6)
model_2 = training(model_2, optimizer_2, criterion, train_loader_2, val_loader_2, num_epochs=num_epochs, device='cuda', min_loss_threshold=0.008)
evaluate(model_2, val_loader_2)
torch.save(model_2.state_dict(), "vit_mushroom_multi_2_best.pth")
#-------------------------------------------------------------------------------------

dataset_augmented_3 = MultiImageMushroomDataset(
    root_dir=train_dir,
    transform=transform_augmented,
    num_images_per_sample=3
)


train_loader_3= DataLoader(
    dataset_augmented_3,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,  # Set to 0 to avoid pickle errors
    pin_memory=True,    # Faster data transfer to GPU
    drop_last=False,          # Use all samples
    persistent_workers=0,  # Keep workers alive between epochs
)

val_loader_3 = DataLoader(
    dataset_augmented_3,
    batch_size=batch_size,
    shuffle=False,  # No need to shuffle validation data
    num_workers=0,
    pin_memory=True,
)

model_3 = ViT_MushroomClassifier(vit_model_name='vit_base_patch16_224', num_classes=num_classes).to(device)
optimizer_3 = torch.optim.AdamW(model_3.parameters(), lr=5e-6)
model_3 = training(model_3, optimizer_3, criterion, train_loader_3, val_loader_3, num_epochs=num_epochs, device='cuda', min_loss_threshold=0.001)
evaluate(model_3, val_loader_3)
torch.save(model_3.state_dict(), "vit_mushroom_multi_3_best.pth")
#-------------------------------------------------------------------------------------
dataset_augmented_4 = MultiImageMushroomDataset(
    root_dir=train_dir,
    transform=transform_augmented,
    num_images_per_sample=4
)


train_loader_4= DataLoader(
    dataset_augmented_4,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,  # Set to 0 to avoid pickle errors
    pin_memory=True,    # Faster data transfer to GPU
    drop_last=False,          # Use all samples
    persistent_workers=0,  # Keep workers alive between epochs
)

val_loader_4 = DataLoader(
    dataset_augmented_4,
    batch_size=batch_size,
    shuffle=False,  # No need to shuffle validation data
    num_workers=0,
    pin_memory=True,
)

model_4 = ViT_MushroomClassifier(vit_model_name='vit_base_patch16_224', num_classes=num_classes).to(device)
optimizer_4 = torch.optim.AdamW(model_4.parameters(), lr=5e-6)
model_4 = training(model_4, optimizer_4, criterion, train_loader_4, val_loader_4, num_epochs=num_epochs, device='cuda', min_loss_threshold=0.001)
evaluate(model_4, val_loader_4)
torch.save(model_4.state_dict(), "vit_mushroom_multi_4_best.pth")
#-------------------------------------------------------------------------------------
dataset_augmented_5 = MultiImageMushroomDataset(
    root_dir=train_dir,
    transform=transform_augmented,
    num_images_per_sample=5
)


train_loader_5= DataLoader(
    dataset_augmented_5,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,  # Set to 0 to avoid pickle errors
    pin_memory=True,    # Faster data transfer to GPU
    drop_last=False,          # Use all samples
    persistent_workers=0,  # Keep workers alive between epochs
)

val_loader_5 = DataLoader(
    dataset_augmented_5,
    batch_size=batch_size,
    shuffle=False,  # No need to shuffle validation data
    num_workers=0,
    pin_memory=True,
)


model_5 = ViT_MushroomClassifier(vit_model_name='vit_base_patch16_224', num_classes=num_classes).to(device)
optimizer_5 = torch.optim.AdamW(model_5.parameters(), lr=5e-6)
model_5 = training(model_5, optimizer_5, criterion, train_loader_5, val_loader_5, num_epochs=num_epochs, device='cuda', min_loss_threshold=0.001)
evaluate(model_5, val_loader_5)
torch.save(model_5.state_dict(), "vit_mushroom_multi_5_best.pth")

#-------------------------------------------------------------------------------------




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
        return Counter(predictions).most_common(1)[0][0]
    elif method == "soft":
        avg_probs = prob_sum / len(models)
        return torch.argmax(avg_probs, dim=1).item()
    else:
        raise ValueError("Unknown ensemble method. Use 'soft' or 'majority'.")


def ensemble_predict_folder_to_csv(models, test_dir, transform, device, method="soft", output_csv="submission_ensemble.csv"):
    results = []
    for filename in sorted(os.listdir(test_dir)):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            path = os.path.join(test_dir, filename)
            predicted_class = ensemble_predict_single_image(
                models=models,
                image_path=path,
                transform=transform,
                device=device,
                method=method
            )
            file_id = os.path.splitext(filename)[0]
            results.append({'id': file_id, 'type': predicted_class})

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Ensemble predictions saved to {output_csv}")




# model_1.load_state_dict(torch.load("D:/IT/GITHUB/Hutech-AI-Challenge/PatternFinding/vit_mushroom_multi_1_best.pth"))
model_2.load_state_dict(torch.load("vit_mushroom_multi_2_best.pth"))
model_3.load_state_dict(torch.load("vit_mushroom_multi_3_best.pth"))
model_4.load_state_dict(torch.load("vit_mushroom_multi_4_best.pth"))
model_5.load_state_dict(torch.load("vit_mushroom_multi_5_best.pth"))


models = [model_2, model_3, model_4, model_5]  # các mô hình đã được load lên device
ensemble_predict_folder_to_csv(
    models=models,
    test_dir=test_dir, #Replace with your test directory path
    transform=transform_original,
    device=device,
    method="soft",  #"soft" hoặc "majority"
    output_csv="submission_ensemble_best_soft_local.csv" #Replace with your desired output file path
)
