import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
from torch.utils.data import ConcatDataset
from torch.utils.data import random_split
import glob
import cv2
import numpy as np
import copy

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

def augment_dataset(dataset, augmentation_transform_list, create_dict = False):
    """
    Augment the dataset with the provided augmentation transforms list.
    :param dataset: the dataset to be augmented
    :param augmentation_transform_list: a list of dictionaries containing the name of the augmentation and the transform
    :param create_dict: whether to create the augmentation_dict or not
    :return: the augmented dataset 
    :return: if create_dict is true,
      a dictionary that maps the name of the augmentation to the index of the transformed dataset in the ConcatDataset.datasets list and the transform
    """
    dataset_list = [dataset]
    if create_dict:
      augmentation_dict = {}
      augmentation_dict[dataset.transform_name] = {"index": 0, "transform": dataset.transform}
    for augmentation_transform_dict in augmentation_transform_list:
      augmentation_name = augmentation_transform_dict["name"]
      augmentation_transform = augmentation_transform_dict["transform"]
      transformed_dataset = CustomImageDataset(dataset.label_file, dataset.imgs_dir, augmentation_transform, augmentation_name, use_cv2=True)
      dataset_list.append(transformed_dataset)
      if create_dict:
        augmentation_dict[augmentation_name] = {"index": len(dataset_list)-1, "transform": augmentation_transform}
    
    augmented_dataset = ConcatDataset(dataset_list)
    if create_dict: return augmented_dataset, augmentation_dict
    return augmented_dataset

def show_augmented_dataset_info(augmented_dataset, augmentation_dict = None):
    """
    Show information of the augmented dataset.
    :param augmented_dataset: the augmented dataset
    :param augmentation_dict: a dictionary that maps the name of the augmentation to the index of the transformed dataset in the ConcatDataset.datasets list and the transform
    """
    assert isinstance(augmented_dataset, ConcatDataset), "Input is not a ConcatDataset"
    if augmentation_dict is not None:
      assert len(augmented_dataset.datasets) == len(augmentation_dict), "Number of augmentations and datasets do not match"
    print("Augmented dataset info:")
    print("Number of augmentations: ", len(augmented_dataset.datasets) - 1)
    if augmentation_dict is not None:
      for key in augmentation_dict:
          print("Transform name: ", key)
          print("Index: ", augmentation_dict[key]["index"])
          print("Transform: ", augmentation_dict[key]["transform"])
          print("Number of images: ", len(augmented_dataset.datasets[augmentation_dict[key]["index"]]))
          print("")
    else:
      for i in range(len(augmented_dataset.datasets)):
          print("Transform name: ", augmented_dataset.datasets[i].transform_name)
          print("Index: ", i)
          print("Transform: ", augmented_dataset.datasets[i].transform)
          print("Number of images: ", len(augmented_dataset.datasets[i]))
          print("")

class CustomImageDataset(Dataset):
    """
    Customized dataset for image classification. It reads image file paths and labels from a csv file and loads
    :param labels_file: path of the csv file containing image names and labels
    :param imgs_dir: directory containing images
    :param transform: transformation to apply to the images
    :param target_transform: transformation to apply to the labels
    :param use_cv2: whether to use cv2 to read images or not
    :return: a dataset object with iterable image dictionary (name and file) and label pairs
    """

    def __init__(self, labels_file, imgs_dir, transform=None, transform_name = "none", target_transform=None, use_cv2=False):
      self.label_file = labels_file
      self.imgs_labels = pd.read_csv(labels_file)
      self.imgs_dir = imgs_dir
      self.transform = transform
      self.transform_name = transform_name
      self.target_transform = target_transform
      self.use_cv2 = use_cv2
      self.cv2_resize = {"resize": True, "resize_height": 256, "resize_width": 256}
      self.adjust_label()
    
    def adjust_label(self):
        # This function removes rows from the dataframe that do not have corresponding image files
        # Create a list of image names
        img_names = [os.path.basename(img) for img in glob.glob(self.imgs_dir + "/*")]
        # Remove dataframe rows with image names not in the list
        self.imgs_labels = self.imgs_labels[self.imgs_labels.iloc[:, 0].isin(img_names)]
        # Check if the number of images is equal to the number of labels
        assert len(self.imgs_labels) == len(os.listdir(self.imgs_dir)), "Number of images and labels do not match"
    
    def set_cv2_resize(self, resize = True, resize_height = 256, resize_width = 256):
        # This function sets the resize parameters if use_cv2 is True
        # Check if use_cv2 is True, else raise an error
        assert self.use_cv2, "Use cv2 is not enabled"
        # Set parameters in cv2_resize
        self.cv2_resize["resize"] = resize
        self.cv2_resize["resize_height"] = resize_height
        self.cv2_resize["resize_width"] = resize_width

    def __len__(self):
      return len(self.imgs_labels)

    def __getitem__(self, idx):
      # Create image path using imgs_dir and img_name from imgs_labels
      img_name = self.imgs_labels.iloc[idx, 0]
      img_path = os.path.join(self.imgs_dir, img_name)
      # Read image file
      if self.use_cv2:
        img_file = np_img_read(img_path, self.cv2_resize["resize"], self.cv2_resize["resize_height"], self.cv2_resize["resize_width"])
      else:
        img_file = read_image(img_path)
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

    def split_train_test(self, anomalous_label = 1, train_ratio = 0.8):
        # Clone the dataset
        train_dataset = copy.deepcopy(self)
        test_dataset = copy.deepcopy(self)
        # TRAIN DATASET
        # Get all data corresponding to non anomalous images
        train_dataset.imgs_labels = train_dataset.imgs_labels[train_dataset.imgs_labels.iloc[:, 1] != anomalous_label]
        # Split the data into train and test
        train_dataset.imgs_labels = train_dataset.imgs_labels.sample(frac = train_ratio)
        # TEST DATASET
        test_dataset.imgs_labels = test_dataset.imgs_labels.drop(train_dataset.imgs_labels.index)
        return train_dataset, test_dataset