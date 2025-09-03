import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import Dataset
import os
import cv2
from PIL import Image

class CatCamDataset(Dataset):
  def __init__(self, root_dir, aug_args, mode="train"):
    self.root_dir = os.path.join(root_dir, mode)
    self.mode = mode
    self.aug_args = aug_args

    self.img_paths = []
    self.labels = []

    for label, class_name in enumerate(sorted(os.listdir(self.root_dir))):
      class_dir = os.path.join(self.root_dir, class_name)

      for img_name in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_name)
        self.img_paths.append(img_path)
        self.labels.append(float(label))

    self._init_transforms()

  def _init_transforms(self):
    if self.mode == "train":
        self.transforms = A.Compose([
            A.HorizontalFlip(p=self.aug_args["hflip_prob"]),
            A.Rotate(
                limit=self.aug_args["rotation_degrees"],
                p=self.aug_args["rotation_prob"]
            ),
            A.Perspective(
                scale=self.aug_args["perspective_scale"],
                p=self.aug_args["perspective_prob"]
            ),
            A.RandomScale(
              scale_limit=self.aug_args["random_scale_limit"], 
              p=self.aug_args["random_scale_prob"]
              ),

            A.PadIfNeeded(
              min_height=self.aug_args["imgsz"], 
              min_width=self.aug_args["imgsz"], 
              border_mode=getattr(cv2, f'BORDER_{self.aug_args["pad_border_mode"].upper()}'), 
              position=self.aug_args["pad_position"],
              p=1.0
            ),

            A.RandomCrop(
              height=self.aug_args["imgsz"], 
              width=self.aug_args["imgsz"], 
              p=1.0
            ),
            
            A.ColorJitter(
                brightness=self.aug_args["brightness_jitter"],
                contrast=self.aug_args["contrast_jitter"],
                saturation=self.aug_args["saturation_jitter"],
                hue=self.aug_args["hue_jitter"],
                p=self.aug_args["color_jitter_prob"]
            ),
            A.ToGray(p=self.aug_args["grayscale_prob"]),
            
            A.GaussianBlur(
                blur_limit=self.aug_args["gaussian_blur_kernel"],
                sigma_limit=self.aug_args["gaussian_blur_sigma"],
                p=self.aug_args["gaussian_blur_prob"]
            ),
            A.GaussNoise(
                var_limit=(0, self.aug_args["gaussian_noise_var"]),
                p=self.aug_args["gaussian_noise_prob"]
            ),
            
            A.CoarseDropout(
                max_holes=self.aug_args["coarse_dropout_max_holes"],
                max_height=self.aug_args["coarse_dropout_max_height"],
                max_width=self.aug_args["coarse_dropout_max_width"],
                min_holes=self.aug_args["coarse_dropout_min_holes"],
                min_height=self.aug_args["coarse_dropout_min_height"],
                min_width=self.aug_args["coarse_dropout_min_width"],
                fill_value=self.aug_args["coarse_dropout_fill_value"],
                p=self.aug_args["coarse_dropout_prob"]
            ),
            
            A.Normalize(
                mean=self.aug_args["normalize_mean"],
                std=self.aug_args["normalize_std"]
            ),
            ToTensorV2()
        ])
    else:
        # Валидация
        self.transforms = A.Compose([
            A.Resize(self.aug_args["imgsz"], self.aug_args["imgsz"]),
            A.Normalize(
                mean=self.aug_args["normalize_mean"],
                std=self.aug_args["normalize_std"]
            ),
            ToTensorV2()
        ])

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()

    img_path = self.img_paths[idx]
    image = Image.open(img_path)
    image = self.transforms(image)
    label = self.labels[idx]

    return (image, label)