import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset

class CustomImageDataset(Dataset):
    """
    Customized dataset for image classification. It reads image file paths and labels from a csv file and loads
    :param labels_file: path of the csv file containing image names and labels
    :param imgs_dir: directory containing images
    :param transform: transformation to apply to the images
    :param target_transform: transformation to apply to the labels
    :return: a dataset object with iterable image dictionary (name and file) and label pairs
    """
    
    def __init__(self, labels_file, imgs_dir, transform=None, target_transform=None):
      self.imgs_labels = pd.read_csv(labels_file)
      self.imgs_dir = imgs_dir
      self.transform = transform
      self.target_transform = target_transform

    def __len__(self):
      return len(self.img_labels)

    def __getitem__(self, idx):
      # Create image path using imgs_dir and img_name from img_labels
      img_name = self.imgs_labels.iloc[idx, 0]
      img_path = os.path.join(self.imgs_dir, img_name)
      # Read image file
      img_file = read_image(img_path)
      # Get image label
      label = self.img_labels.iloc[idx, 1]
      # Apply transformations
      if self.transform:
        img_file = self.transform(img_file)
      if self.target_transform:
        label = self.target_transform(label)
      # Build image dictionary and return
      image = {"name": self.img_labels.iloc[idx,0],
               "file": img_file}
      return image, label