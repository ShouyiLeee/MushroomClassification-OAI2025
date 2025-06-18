import glob
import os
import re

import torch
from PIL import Image
from torch.utils.data import Dataset


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
