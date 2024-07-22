import json
import os
import pandas as pd


# Parameters
dataset_dir = './dataset/to_combine/'
output_dir = './dataset/combined/'
output_images_dir = output_dir + 'images/'
dataset_info_file_name = 'datasets_info.json'

# Create the DataFrame that will contain all the data
df = pd.DataFrame(columns=['filename', 'label'])

# Load json file that describe all the datasets
dataset_info_file = os.path.join(dataset_dir, dataset_info_file_name)
with open(dataset_dir + 'datasets_info.json') as f:
    datasets_info = json.load(f)

path_dict = {}
for dataset in datasets_info["datasets"]:
    # Extract dataset information
    dataset_name = dataset["name"]
    dataset_type = dataset["type"]
    dataset_label_file = dataset["label_file"]
    dataset_filename_with_path = dataset["filename_with_path"]
    print('Processing dataset:', dataset_name)
    print('Type:', dataset_type)
    print('Label file:', dataset_label_file)
    print('Filename with path:', dataset_filename_with_path)

    # Get dataset picture filenames (iterate in each directory)
    images_list = []
    path_list = []
    for path, subdirs, files in os.walk(os.path.join(dataset_dir, dataset_name)):
        if files:
            # If required add the relative path of the images
            if dataset_filename_with_path:
                path_separator = dataset_name + '/'
                relative_path = path.split(path_separator)[1]
                path_list.append(relative_path)
                files = [os.path.join(relative_path, file) for file in files]
            images_list = images_list + files
    path_dict[dataset_name] = path_list
    print('Number of images:', len(images_list))
    print('First image:', images_list[0])

    # Assign labels to the images
    # TODO: Implement function to assign label using the label file if provided and if the dataset is "mixed"
    if dataset_type == "non-anomalous":
        label = 0
    elif dataset_type == "anomalous":
        label = 1

    # Build dictionaries list with the images and labels
    images_dict_list = []
    for image in images_list:
        images_dict = {"filename": image, "label": label}
        images_dict_list.append(images_dict)
    
    # Convert the dictionaries list to a DataFrame
    df_to_append = pd.DataFrame.from_dict(images_dict_list)

    # Concatenate the DataFrames
    df = pd.concat([df, df_to_append], ignore_index=True)
    
# Show info of the DataFrame
print(df.head())
print(df.tail())
print(df['label'].value_counts())

# Save the DataFrame to a csv file
df.to_csv(output_dir + 'combined_dataset.csv', index=False)

# Execute the bash script to create the dataset directory structure
for dataset in datasets_info["datasets"]:
    # Extract dataset information
    dataset_name = dataset["name"]
    dataset_filename_with_path = dataset["filename_with_path"]
    dataset_full_dir = os.path.join(dataset_dir, dataset_name)
    if dataset_filename_with_path:
        for path in path_dict[dataset_name]:
            input_file_path = os.path.join(dataset_full_dir, path)
            output_file_path = os.path.join(output_images_dir, path)
            os.system('mkdir -p ' + output_file_path)
            os.system('bash ./create_dataset.sh ' + input_file_path + ' ' + output_file_path)
    else:
        os.system('bash ./create_dataset.sh ' + dataset_full_dir + ' ' + output_images_dir)