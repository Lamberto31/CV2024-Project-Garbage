import argparse
import os
import torch
import torch.utils.data as data
import torchvision.transforms as T
import torch.nn as nn
import json

from data.CustomDataset import CustomImageDataset
from data.CustomDataset import augment_dataset
from data.CustomDataset import show_augmented_dataset_info
from evaluation_utils import *

def classify(image_file_path):
    # PARAMETERS
    args_dict = {
    "gpus": "1",                            # gpus (set 1 or None)
    "batch_size": 1,                        # batch size for testing
    "method": "recon",                      # The target task for anoamly detection  (pred or recon)
    "fdim": 512,                            # channel dimension of the features
    "mdim": 512,                            # channel dimension of the memory items
    "alpha": 0.7,                           # weight for the anomality score
    "th": 0.015,                            # threshold for test updating
    "threshold": 0.36,                      # threshold for the anomaly score
    "num_workers": 1,                       # number of workers for the test loader
    "dataset_type": "ade_20k_and_taco",     # type of dataset: clean_road, ade_20k_and_taco
    "model_path": "./model/trained",        # directory of model
    "model_file": "model.pth",              # name of the model file
    "m_items_path": "./model/trained",      # directory of memory items
    "m_items_file": "keys.pt",              # name of the memory items file
    "augment": False,                       # whether to use data augmentation
    "use_custom_min_max": False,            # use custom min and max values for normalization
    "custom_min_max_file": "min_max.json"   # file with custom min and max for values for normalization
    }
    args = argparse.Namespace(**args_dict)



    # GPU CONFIGURATIONS
    #print(torch.cuda.is_available())
    if not torch.cuda.is_available():
        print("CUDA is not available.  Exiting ...")
        exit()

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    if args.gpus is None:
        gpus = "0"
        os.environ["CUDA_VISIBLE_DEVICES"]= gpus
    else:
        gpus = ""
        for i in range(len(args.gpus)):
            gpus = gpus + args.gpus[i] + ","
        os.environ["CUDA_VISIBLE_DEVICES"]= gpus[:-1]

    #torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance



    # DATA LOADING
    # TODO: Assign the correct paths when called as function
    # TEMP
    test_folder = "./dataset/single/images"
    test_label_folder = "./dataset/single/"
    test_label_file = os.path.join(test_label_folder, "test_labels.csv")
    # TEMP_END

    #transform = T.Resize((args.h,args.w))
    transform = T.Compose([T.ToTensor(),])

    # Loading dataset
    test_dataset = CustomImageDataset(test_label_file, test_folder, transform = transform, use_cv2=True)
    test_size = len(test_dataset)

    test_batch = data.DataLoader(test_dataset, batch_size = args.batch_size,
                                shuffle=True, num_workers=args.num_workers, drop_last=False)
    batch_size = len(test_batch)

    # Load custom min and max values for normalization if required
    if args.use_custom_min_max:
        # TEMP
        min_max_file = os.path.join("./dataset/single/", args.custom_min_max_file)
        # TEMP_END
        with open(min_max_file, 'r') as f:
            min_max = json.load(f)
    else:
        min_max = None


    # DATA AUGMENTATION
    if args.augment:
        # Create augmentation transform list
        augmentation_transform_list = []

        enabled = True
        if enabled:
            augmentation_transform = T.Compose([
                T.ToPILImage(),
                T.RandAugment(),
                T.ToTensor(),    
            ])
            transform_name = "RandAugment"
            applications_number = 3
            transform_dict = {"name": transform_name, "transform": augmentation_transform, "applications_number": applications_number}
            augmentation_transform_list.append(transform_dict)


        # Apply augment_dataset function to create augmented dataset
        augmented_test_dataset = augment_dataset(test_dataset, augmentation_transform_list, create_dict=False)
        augmented_test_size = len(augmented_test_dataset)

        augmented_test_batch = data.DataLoader(augmented_test_dataset, batch_size = args.batch_size,
                                    shuffle=True, num_workers=args.num_workers, drop_last=False)
        augmented_batch_size = len(augmented_test_batch)

        test_batch = augmented_test_batch

        #show_augmented_dataset_info(augmented_test_dataset)



    # MODEL LOADING
    # Loading the trained model
    model_file = os.path.join(args.model_path, args.dataset_type, args.model_file)
    m_items_file = os.path.join(args.m_items_path, args.dataset_type, args.m_items_file)
    model = torch.load(model_file)
    model.cuda()
    m_items = torch.load(m_items_file)



    # EVALUATION SETUP
    loss_func_mse = nn.MSELoss(reduction='none')

    psnr_list = {}
    feature_distance_list = {}

    # Populate the dictionaries with the image names and empty lists
    for img_name in test_dataset.imgs_labels["filename"].to_numpy():
        psnr_list[img_name] = []
        feature_distance_list[img_name] = []

    m_items_test = m_items.clone()


    # EVALUATION
    model.eval()

    for j,(images, labels) in enumerate(test_batch):
        imgs = images["file"]
        img_name = images["name"][0]

        if args.gpus is not None and torch.cuda.is_available():
            imgs = imgs.cuda()

        if args.method == 'pred':
            outputs, feas, updated_feas, m_items_test, softmax_score_query, softmax_score_memory, _, _, _, compactness_loss = model.forward(imgs[:,0:3*4], m_items_test, False)
            mse_imgs = torch.mean(loss_func_mse((outputs[0]+1)/2, (imgs[0,3*4:]+1)/2)).item()
            mse_feas = compactness_loss.item()

            # Calculating the threshold for updating at the test time
            point_sc = point_score(outputs, imgs[:,3*4:])

        else:
            outputs, feas, updated_feas, m_items_test, softmax_score_query, softmax_score_memory, compactness_loss = model.forward(imgs, m_items_test, False)
            mse_imgs = torch.mean(loss_func_mse((outputs[0]+1)/2, (imgs[0]+1)/2)).item()
            mse_feas = compactness_loss.item()

            # Calculating the threshold for updating at the test time
            point_sc = point_score(outputs, imgs)

        if  point_sc < args.th:
            query = nn.functional.normalize(feas, dim=1)
            query = query.permute(0,2,3,1) # b X h X w X d
            m_items_test = model.memory.update(query, m_items_test, False)
        
        psnr_list[img_name].append(psnr(mse_imgs))
        feature_distance_list[img_name].append(mse_feas)

    # Measuring the abnormality score and the AUC
    anomaly_score_total_list = []

    # Calculating the abnormality score as the sum of the PSNR (inverted) and the feature distance
    psnr_listed = list(psnr_list.values())
    feature_distance_listed = list(feature_distance_list.values())
    if args.use_custom_min_max:
        psnr_listed.extend([min_max["psnr_min"], min_max["psnr_max"]])
        feature_distance_listed.extend([min_max["feature_distance_min"], min_max["feature_distance_max"]])
    psnr_listed = anomaly_score_list_inv(psnr_listed)
    feature_distance_listed = anomaly_score_list(feature_distance_listed)

    # Remove the added values from the two list if custom min and max values are used
    if args.use_custom_min_max:
        psnr_listed = psnr_listed[:-2]
        feature_distance_listed = feature_distance_listed[:-2]
        
    anomaly_score_total_list = score_sum(psnr_listed, feature_distance_listed, args.alpha)

    anomaly_score_total_list = np.asarray(anomaly_score_total_list)
    anomaly_score_total_list = np.mean(anomaly_score_total_list, axis=1)
    anomaly_score_total_list = np.expand_dims(anomaly_score_total_list, axis=0).T

    #print(anomaly_score_total_list.shape)
    #print(anomaly_score_total_list)

    # Calculating the AUC
    anomaly_score_total_list = np.expand_dims(anomaly_score_total_list, axis=0)
    labels_list = np.expand_dims(test_dataset.imgs_labels["label"].to_numpy(), axis=0)


    # Classification
    thresholded_score_list, _, _, _, _, _ = classify_with_threshold(anomaly_score_total_list, labels_list, args.threshold)
    print("Anomaly score:")
    print(np.squeeze(anomaly_score_total_list))
    print("Thresholded score:")
    print(np.squeeze(thresholded_score_list))

    # Return
    if thresholded_score_list == 1:
        return 1
    else:
        return 0
