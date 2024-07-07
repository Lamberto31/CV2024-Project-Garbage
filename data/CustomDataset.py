import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
import glob
import cv2
import numpy as np

def np_img_read(filename, resize = True, resize_height = 256, resize_width = 256):
    """
    Load image and convert it to numpy.ndarray. Notes that the color channels are BGR and the color space
    is normalized from [0, 255] to [-1, 1].

    :param filename: the full path of image
    :param resize: whether to resize the image or not
    :param resize_height: resized height
    :param resize_width: resized width
    :return: numpy.ndarray
    """

    image_decoded = cv2.imread(filename)
    if resize:
        image_decoded = cv2.resize(image_decoded, (resize_width, resize_height))
    image_decoded = image_decoded.astype(dtype=np.float32)
    image_decoded = (image_decoded / 127.5) - 1.0
    return image_decoded

def create_img_to_show(img_file, show_shapes=False):
    """
    Prepare an image to be shown with matplotlib. The output is a numpy.ndarray with RGB color space.
    The provided img_file is a ready to use tensor for the model (as input or output).
    :param img_file: the image tensor. Shape: (C, H, W), color space: BGR, normalized: [-1, 1]
    :param show_shapes: whether to print the shapes of the image tensor or not
    :return: numpy.ndarray
    """
    # Create img_to_show
    img_to_show = img_file
    # Permute the image from (C, H, W) to (H, W, C)
    if show_shapes: print(img_to_show.shape)
    img_to_show = img_to_show.permute(1, 2, 0)
    if show_shapes: print(img_to_show.shape)
    # Convert the image from tensor to numpy array
    img_to_show = img_to_show.numpy()
    # Adapt BGR to RGB for matplotlib
    img_to_show = cv2.cvtColor(img_to_show, cv2.COLOR_BGR2RGB)
    return img_to_show

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
      self.adjust_label()
    
    def adjust_label(self):
        # This function removes rows from the dataframe that do not have corresponding image files
        # Create a list of image names
        img_names = [os.path.basename(img) for img in glob.glob(self.imgs_dir + "/*")]
        # Remove dataframe rows with image names not in the list
        self.imgs_labels = self.imgs_labels[self.imgs_labels.iloc[:, 0].isin(img_names)]
        # Check if the number of images is equal to the number of labels
        assert len(self.imgs_labels) == len(os.listdir(self.imgs_dir)), "Number of images and labels do not match"


    def __len__(self):
      return len(self.imgs_labels)

    def __getitem__(self, idx):
      # Create image path using imgs_dir and img_name from imgs_labels
      img_name = self.imgs_labels.iloc[idx, 0]
      img_path = os.path.join(self.imgs_dir, img_name)
      # Read image file
      #img_file = read_image(img_path)
      img_file = np_img_read(img_path)
      # Get image label
      label = self.imgs_labels.iloc[idx, 1]
      # Apply transformations
      if self.transform:
        img_file = self.transform(img_file)
      if self.target_transform:
        label = self.target_transform(label)
      # Build image dictionary and return
      image = {"name": self.imgs_labels.iloc[idx,0],
               "file": img_file}
      return image, label
    
    def get_img_by_name(self, img_name):
        # Get image by name
        img_path = os.path.join(self.imgs_dir, img_name)
        # Read image file
        img_file = np_img_read(img_path)
        # Get image label
        label = self.imgs_labels[self.imgs_labels.iloc[:, 0] == img_name].iloc[0, 1]
        # Apply transformations
        if self.transform:
          img_file = self.transform(img_file)
        if self.target_transform:
          label = self.target_transform(label)
        return img_file