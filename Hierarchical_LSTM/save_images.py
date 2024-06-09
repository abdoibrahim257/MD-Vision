import os
import cv2
import numpy as np
from pathlib import Path
import argparse
from sklearn.model_selection import train_test_split

def main(args):
    # Get the path to the Indiana dataset
    curr_path = Path(os.getcwd())
    Indiana_path =  os.path.join(str(curr_path), 'Data')
    
    # Load the image names
    images = os.listdir(f'{Indiana_path}/Images')
    
    # Split the data into training and validation sets
    images_train, images_val = train_test_split(images, test_size = 0.15, random_state = 42)

    # Process training images
    images_train = process_images(args, Indiana_path, images_train)
    np.save('./Data/images_train.npy', images_train)

    # Process validation images
    images_val = process_images(args, Indiana_path, images_val)
    np.save('./Data/images_val.npy', images_val)

def process_images(args, path, image_list):
    # Load the images from the path
    images = [cv2.imread(f'{path}/Images/{image}') for image in image_list]
    # Resize the images to the specified size
    images = [cv2.resize(image, (args.img_size, args.img_size)) for image in images]
    # Center crop
    center = args.img_size//2, args.img_size//2
    x = center[0] - args.crop_size//2
    y = center[1] - args.crop_size//2
    images = [image[x:x+args.crop_size, y:y+args.crop_size] for image in images]
    # Convert the images to grayscale
    images = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]
    # Convert the images to a numpy array
    images = np.array(images)

    return images

if __name__ == "__main__":
    # Define args here
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_size', type=int, default=256, help='Size of the images')
    parser.add_argument('--crop_size', type=int, default=224, help='Size of the center crop')
    args = parser.parse_args()
    main(args)