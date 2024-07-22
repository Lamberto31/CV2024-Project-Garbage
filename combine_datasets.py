import json
import os
import pandas as pd


# Parameters
dataset_dir = './dataset/to_combine/'
output_dir = './dataset/combined/'
dataset_info_file_name = 'datasets_info.json'

# Create the DataFrame that will contain all the data
df = pd.DataFrame(columns=['filename', 'label'])

# Load json file that describe all the datasets
dataset_info_file = os.path.join(dataset_dir, dataset_info_file_name)
with open(dataset_dir + 'datasets_info.json') as f:
    datasets_info = json.load(f)

for dataset in datasets_info["datasets"]:
    # Extract dataset information
    dataset_name = dataset["name"]
    dataset_type = dataset["type"]
    dataset_label_file = dataset["label_file"]
    print('Processing dataset:', dataset_name)
    print('Type:', dataset_type)
    print('Label file:', dataset_label_file)

    # Get dataset picture filenames (iterate in each directory)
    images_list = []
    for path, subdirs, files in os.walk(os.path.join(dataset_dir, dataset_name)):
        if files:
            images_list = images_list + files
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