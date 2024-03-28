import os
import sys
import json
import glob
import time
import shutil
import inspect
import warnings
import argparse
import traceback
from tqdm import tqdm

import cv2
import math
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from ultralytics import YOLO
from ultralytics.utils import checks

from tensorboard import program

from Common import get_now

warnings.filterwarnings('ignore')


# ------------------------------------------------------------------------------------------------------------------
# Functions
# ------------------------------------------------------------------------------------------------------------------
def get_classifier_encoders():
    """
    Lists all the models available
    """

    return [f"yolo8{_}-cls.pt" for _ in ['n', 's', 'm', 'l', 'x']]


def compute_class_weights(df, mu=0.15):
    """
    Compute class weights for the given dataframe.
    """
    # Compute the value counts for each class
    class_categories = sorted(df['Label'].unique())
    class_values = [df['Label'].value_counts()[c] for c in class_categories]
    value_counts = dict(zip(class_categories, class_values))

    total = sum(value_counts.values())
    keys = value_counts.keys()

    # To store the class weights
    class_weight = dict()

    # Compute the class weights for each class
    for key in keys:
        score = math.log(mu * total / float(value_counts[key]))
        class_weight[key] = score if score > 1.0 else 1.0

    return class_weight


# ------------------------------------------------------------------------------------------------------------------
# Training
# ------------------------------------------------------------------------------------------------------------------

def classification(args):
    """

    """
    print("\n###############################################")
    print(f"Classification")
    print("###############################################\n")

    # Check for CUDA
    print(f"NOTE: Ultralytics version - {checks.check_version()}")
    print(f"NOTE: YOLO version - {checks.check_yolo()}")
    print(f"NOTE: CUDA count - {checks.cuda_device_count()}")

    # Whether to run on GPU or CPU
    device = 'cuda' if checks.cuda_is_available() else 'cpu'

    # ------------------------------------------------------------------------------------------------------------------
    # Check input data
    # ------------------------------------------------------------------------------------------------------------------
    # TODO this will likely need to be modified as the user is bringing in data from Classifier Widget
    #   Make this into a function that converts multiple CSV files and patch datasets into a .yaml file

    # If the user provides multiple patch dataframes
    patches_df = pd.DataFrame()

    for patches_path in args.patches:
        if os.path.exists(patches_path):
            # Patch dataframe
            patches = pd.read_csv(patches_path, index_col=0)
            patches = patches.dropna()
            patches_df = pd.concat((patches_df, patches))
        else:
            raise Exception(f"ERROR: Patches dataframe {patches_path} does not exist")

    class_names = sorted(patches_df['Label'].unique())
    num_classes = len(class_names)
    class_map = {f"{class_names[_]}": _ for _ in range(num_classes)}

    # ------------------------------------------------------------------------------------------------------------------
    # Building Model
    # ------------------------------------------------------------------------------------------------------------------
    print(f"\n#########################################\n"
          f"Building Model\n"
          f"#########################################\n")

    try:
        # Make sure it's a valid choice
        if args.encoder_name not in get_classifier_encoders():
            raise Exception(f"ERROR: Encoder must be one of {get_classifier_encoders()}")

        # Pre-trained is either True, False, or a path to weights
        pretrained = True

        if type(args.pre_trained) == bool:
            pretrained = args.pre_trained
        elif type(args.pre_trained) == str and os.path.exists(args.pre_trained):
            pretrained = args.pre_trained
        print(f"NOTE: Using pre-trained weights - {pretrained}")

        # Use the encoder specified
        model = YOLO(args.encoder_name)
        print(f"NOTE: Using {args.encoder_name} encoder")

        # TODO "freeze" the first N layers
        # Freezing percentage of the encoder
        num_params = len(list(model.encoder.parameters()))
        freeze_params = int(num_params * args.freeze_encoder)

        # Give users the ability to freeze N percent of the encoder
        print(f"NOTE: Freezing {args.freeze_encoder}% of encoder weights")
        for idx, param in enumerate(model.encoder.parameters()):
            if idx < freeze_params:
                param.requires_grad = False

    except Exception as e:
        raise Exception(f"ERROR: Could not build model\n{e}")

    # ---------------------------------------------------------------------------------------
    # Source directory setup
    # ---------------------------------------------------------------------------------------
    print("\n###############################################")
    print("Logging")
    print("###############################################\n")

    output_dir = f"{args.output_dir}/"

    # Run Name
    run = f"{get_now()}_{args.encoder_name}"

    # We'll also create folders in this source to hold results of the model
    run_dir = f"{output_dir}classification/{run}/"
    weights_dir = run_dir + "weights/"
    logs_dir = run_dir + "logs/"
    tensorboard_dir = logs_dir + "tensorboard/"

    # Make the directories
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)

    with open(f'{run_dir}Class_Map.json', 'w') as json_file:
        json.dump(class_map, json_file, indent=3)

    print(f"NOTE: Model Run - {run}")
    print(f"NOTE: Model Directory - {run_dir}")
    print(f"NOTE: Log Directory - {logs_dir}")
    print(f"NOTE: Tensorboard Directory - {tensorboard_dir}")

    # Open tensorboard
    if args.tensorboard:
        print(f"\n#########################################\n"
              f"Tensorboard\n"
              f"#########################################\n")

        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', tensorboard_dir])
        url = tb.launch()

        print(f"NOTE: View Tensorboard at {url}")

    # ------------------------------------------------------------------------------------------------------------------
    # Loading data, creating datasets
    # ------------------------------------------------------------------------------------------------------------------
    print("\n###############################################")
    print("Creating Datasets")
    print("###############################################\n")
    # Names of all images; sets to be split based on images
    image_names = patches_df['Image Name'].unique()

    # Split the Images into training, validation, and test sets (70/20/10)
    # We split based on the image names, so that we don't have the same image in multiple sets.
    training_images, temp_images = train_test_split(image_names, test_size=0.3, random_state=42)
    validation_images, testing_images = train_test_split(temp_images, test_size=0.33, random_state=42)

    # Create training, validation, and test dataframes
    train_df = patches_df[patches_df['Image Name'].isin(training_images)]
    valid_df = patches_df[patches_df['Image Name'].isin(validation_images)]
    test_df = patches_df[patches_df['Image Name'].isin(testing_images)]

    train_classes = len(set(train_df['Label'].unique()))
    valid_classes = len(set(valid_df['Label'].unique()))
    test_classes = len(set(test_df['Label'].unique()))

    # If there isn't one class sample in each data sets
    # will throw an error; hacky way of fixing this.
    if not (train_classes == valid_classes == test_classes):
        print("NOTE: Sampling one of each class category")
        # Holds one sample of each class category
        sample = pd.DataFrame()
        # Gets one sample from patches_df
        for label in patches_df['Label'].unique():
            one_sample = patches_df[patches_df['Label'] == label].sample(n=1)
            sample = pd.concat((sample, one_sample))

        train_df = pd.concat((sample, train_df))
        valid_df = pd.concat((sample, valid_df))
        test_df = pd.concat((sample, test_df))

    train_df.reset_index(drop=True, inplace=True)
    valid_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    # TODO make this into a YAML file for training
    #   The user will provide N CSV files, here we'll create a YAML for training

    # Output to logs
    train_df.to_csv(f"{logs_dir}Training_Set.csv", index=False)
    valid_df.to_csv(f"{logs_dir}Validation_Set.csv", index=False)
    test_df.to_csv(f"{logs_dir}Testing_Set.csv", index=False)

    # The number of class categories
    print(f"NOTE: Number of classes in training set is {len(train_df['Label'].unique())}")
    print(f"NOTE: Number of classes in validation set is {len(valid_df['Label'].unique())}")
    print(f"NOTE: Number of classes in testing set is {len(test_df['Label'].unique())}")

    # ------------------------------------------------------------------------------------------------------------------
    # Data Exploration
    # ------------------------------------------------------------------------------------------------------------------

    plt.figure(figsize=(10, 5))

    # Set the same y-axis limits for all subplots
    ymin = 0
    ymax = train_df['Label'].value_counts().max() + 10

    # Plotting the train data
    plt.subplot(1, 3, 1)
    plt.title(f"Train: {len(train_df)} Classes: {len(train_df['Label'].unique())}")
    ax = train_df['Label'].value_counts().plot(kind='bar')
    ax.set_ylim([ymin, ymax])

    # Plotting the valid data
    plt.subplot(1, 3, 2)
    plt.title(f"Valid: {len(valid_df)} Classes: {len(valid_df['Label'].unique())}")
    ax = valid_df['Label'].value_counts().plot(kind='bar')
    ax.set_ylim([ymin, ymax])

    # Plotting the test data
    plt.subplot(1, 3, 3)
    plt.title(f"Test: {len(test_df)} Classes: {len(test_df['Label'].unique())}")
    ax = test_df['Label'].value_counts().plot(kind='bar')
    ax.set_ylim([ymin, ymax])

    # Saving and displaying the figure
    plt.savefig(logs_dir + "DatasetSplit.png")
    plt.close()

    if os.path.exists(logs_dir + "DatasetSplit.png"):
        print(f"NOTE: Datasplit Figure saved in {logs_dir}")

    # ------------------------------------------------------------------------------------------------------------------
    # Dataset creation
    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    # Model Training
    # ------------------------------------------------------------------------------------------------------------------



# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Train an Image Classifier')

    parser.add_argument('--patches', required=True, nargs="+",
                        help='The path to the patch labels csv file output the Patches tool')

    parser.add_argument('--pre_trained_path', type=str, default=None,
                        help='Path to pre-trained model of the same architecture')

    parser.add_argument('--encoder_name', type=str, default='efficientnet-b0',
                        help='The convolutional encoder to fine-tune; pretrained on Imagenet')

    parser.add_argument('--freeze_encoder', type=float, default=0.0)

    parser.add_argument('--encoder_name', type=str, default='efficientnet-b0',
                        help='The convolutional encoder to fine-tune; pretrained on Imagenet')

    parser.add_argument('--freeze_encoder', type=float,
                        help='Freeze N% of the encoder [0 - 1]')

    parser.add_argument('--loss_function', type=str, default='CrossEntropyLoss',
                        help='The loss function to use to train the model')

    parser.add_argument('--weighted_loss', type=bool, default=True,
                        help='Use a weighted loss function; good for imbalanced datasets')

    parser.add_argument('--metrics', nargs="+", default=[],
                        help='The metrics used to evaluate the model (default is all applicable)')

    parser.add_argument('--optimizer', default="Adam",
                        help='Optimizer for training the model')

    parser.add_argument('--learning_rate', type=float, default=0.0005,
                        help='Initial learning rate')

    parser.add_argument('--augment_data', action='store_true',
                        help='Apply affine augmentations to training data')

    parser.add_argument('--dropout_rate', type=float, default=0.5,
                        help='Amount of dropout in model (augmentation)')

    parser.add_argument('--num_epochs', type=int, default=5,
                        help='Starting learning rate')

    parser.add_argument('--batch_size', type=int, default=128,
                        help='Starting learning rate')

    parser.add_argument('--tensorboard', action='store_true',
                        help='Display training on Tensorboard')

    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save updated label csv file.')

    args = parser.parse_args()

    try:
        classification(args)
        print("Done.\n")

    except Exception as e:
        print(f"ERROR: {e}")
        print(traceback.format_exc())


if __name__ == '__main__':
    main()