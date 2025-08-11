import torchvision.transforms as T
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
from PIL import Image
import requests
from io import BytesIO

def create_url_df(cat_urls_path: str, no_cat_urls_path: str) -> pd.DataFrame:
  with open(cat_urls_path) as f:
    cat_urls = f.readlines()

  with open(no_cat_urls_path) as f:
    no_cat_urls = f.readlines()

  urls = np.concatenate([cat_urls, no_cat_urls])
  labels = np.concatenate([np.ones_like(cat_urls, dtype=np.float64), np.zeros_like(no_cat_urls, dtype=np.float64)])
  url_df = pd.DataFrame({
      'url' : urls,
      'label' : labels,
  })

  return url_df

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
        self.labels.append(float(label-1))

    self._init_transforms()

  def _init_transforms(self):
    if (self.mode == "train"):
      self.transforms = T.Compose(
          [
            T.Resize((self.aug_args["imgsz"], self.aug_args["imgsz"])),
            T.RandomHorizontalFlip(p=self.aug_args["hflip_prob"]),
            T.RandomRotation(self.aug_args["rotation_degrees"]),
            T.RandomPerspective(distortion_scale=self.aug_args["perspective_scale"], p=self.aug_args["perspective_prob"]),
            T.ColorJitter(
                brightness=self.aug_args["brightness_jitter"],
                contrast=self.aug_args["contrast_jitter"],
                saturation=self.aug_args["saturation_jitter"],
                hue=self.aug_args["hue_jitter"]
            ),
            T.RandomGrayscale(p=self.aug_args["grayscale_prob"]),
            T.GaussianBlur(
                kernel_size=self.aug_args["gaussian_blur_kernel"],
                sigma=self.aug_args["gaussian_blur_sigma"]
            ),
            T.ToTensor(),
            T.Normalize(
                mean=self.aug_args["normalize_mean"],
                std=self.aug_args["normalize_std"]
            ),
            T.RandomErasing(
                p=self.aug_args["random_erase_prob"],
                scale=self.aug_args["random_erase_scale"]
            )
        ])
    else:
      self.transforms = T.Compose(
          [
            T.Resize((self.aug_args["imgsz"], self.aug_args["imgsz"])),
            T.CenterCrop(self.aug_args["center_crop_size"]),
            T.ToTensor(),
            T.Normalize(mean=self.aug_args["normalize_mean"], std=self.aug_args["normalize_std"])
          ]
        )

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