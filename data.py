# Import necessary libraries
import torch
import torch.utils.data as data
import csv
import os, math
import sys
import time
import random
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Dataset class for CrackerBox detection
class CrackerBox(data.Dataset):
    def __init__(self, image_set='train', data_path='homework2_programming_Rahil/yolo/data'):
        # Set dataset name and paths
        self.name = 'cracker_box_' + image_set
        self.image_set = image_set
        self.data_path = data_path

        # Define class names
        self.classes = ('__background__', 'cracker_box')

        # Original and target image dimensions
        self.width = 640
        self.height = 480
        self.yolo_image_size = 448  # YOLO input size

        # Compute scaling factors
        self.scale_width = self.yolo_image_size / self.width
        self.scale_height = self.yolo_image_size / self.height

        # YOLO grid parameters
        self.yolo_grid_num = 7
        self.yolo_grid_size = self.yolo_image_size / self.yolo_grid_num

        # Get file paths for ground truth annotations
        self.gt_files_train, self.gt_files_val = self.list_dataset()

        # Mean pixel values for normalization
        self.pixel_mean = np.array([[[102.9801, 115.9465, 122.7717]]], dtype=np.float32)

        # Assign paths based on train/val split
        if image_set == 'train':
            self.size = len(self.gt_files_train)
            self.gt_paths = self.gt_files_train
            print('%d images for training' % self.size)
        else:
            self.size = len(self.gt_files_val)
            self.gt_paths = self.gt_files_val
            print('%d images for validation' % self.size)

    # Function to list annotation (.txt) files
    def list_dataset(self):
        filename = os.path.join(self.data_path, '*.txt')
        gt_files = sorted(glob.glob(filename))  # Sort to ensure deterministic order
        print("Looking for txt files in:", os.path.abspath(self.data_path))
        return gt_files[:100], gt_files[100:]

    # Main function to load and process each sample
    def __getitem__(self, idx):
        filename_gt = self.gt_paths[idx]
        image_file = filename_gt.replace('-box.txt', '.jpg')  # Get corresponding image

        # Load image
        image = cv2.imread(image_file)
        if image is None:
            raise FileNotFoundError(f"Image file {image_file} not found.")

        # Resize, normalize, and reformat image
        image = cv2.resize(image, (self.yolo_image_size, self.yolo_image_size))
        image = image.astype(np.float32) - self.pixel_mean  # Subtract mean
        image = image / 255.0
        image = image.transpose((2, 0, 1))  # Channels first
        image_blob = torch.from_numpy(image).float()

        # Load bounding box (x1, y1, x2, y2)
        bbox = np.loadtxt(filename_gt)
        x1, y1, x2, y2 = bbox

        # Scale bbox to resized image dimensions
        x1 *= self.scale_width
        y1 *= self.scale_height
        x2 *= self.scale_width
        y2 *= self.scale_height

        # Convert to YOLO format: center x/y, width, height
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = x2 - x1
        h = y2 - y1

        cx_norm = cx / self.yolo_grid_size
        cy_norm = cy / self.yolo_grid_size
        w_norm = w / self.yolo_image_size
        h_norm = h / self.yolo_image_size

        # Determine grid cell index and offset
        grid_x = int(cx_norm)
        grid_y = int(cy_norm)
        cx_offset = cx_norm - grid_x
        cy_offset = cy_norm - grid_y

        # Create output tensors for YOLO
        gt_box_blob = torch.zeros(5, self.yolo_grid_num, self.yolo_grid_num)
        gt_mask_blob = torch.zeros(self.yolo_grid_num, self.yolo_grid_num)

        # Populate the responsible grid cell with values
        gt_box_blob[:, grid_y, grid_x] = torch.tensor([cx_offset, cy_offset, w_norm, h_norm, 1.0])
        gt_mask_blob[grid_y, grid_x] = 1.0

        # Return sample dictionary
        return {
            'image': image_blob,
            'gt_box': gt_box_blob,
            'gt_mask': gt_mask_blob
        }

    # Return dataset length
    def __len__(self):
        return self.size

# Function to draw a yellow grid over an image (used in visualization)
def draw_grid(image, line_space=64):
    H, W = image.shape[:2]
    image[0:H:line_space] = [255, 255, 0]  # Horizontal lines
    image[:, 0:W:line_space] = [255, 255, 0]  # Vertical lines

# Main testing script
if __name__ == '__main__':
    # Load training and validation datasets
    dataset_train = CrackerBox('train')
    dataset_val = CrackerBox('val')

    # Create a DataLoader to iterate over training samples
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=1, shuffle=False, num_workers=0)

    # Visualize data samples
    for i, sample in enumerate(train_loader):
        image = sample['image'][0].numpy().transpose((1, 2, 0))
        gt_box = sample['gt_box'][0].numpy()
        gt_mask = sample['gt_mask'][0].numpy()

        # Get grid cell where object is located
        y, x = np.where(gt_mask == 1)

        # Decode YOLO format back to pixel coordinates
        cx = gt_box[0, y, x] * dataset_train.yolo_grid_size + x * dataset_train.yolo_grid_size
        cy = gt_box[1, y, x] * dataset_train.yolo_grid_size + y * dataset_train.yolo_grid_size
        w = gt_box[2, y, x] * dataset_train.yolo_image_size
        h = gt_box[3, y, x] * dataset_train.yolo_image_size

        x1 = cx - w * 0.5
        x2 = cx + w * 0.5
        y1 = cy - h * 0.5
        y2 = cy + h * 0.5

        # Reconstruct image
        im = image * 255.0 + dataset_train.pixel_mean
        im = im.astype(np.uint8)

        # Plot original image
        fig = plt.figure()
        ax = fig.add_subplot(1, 3, 1)
        plt.imshow(im[:, :, (2, 1, 0)])
        plt.title('input image (448x448)', fontsize=16)

        # Plot image with grid and bbox
        ax = fig.add_subplot(1, 3, 2)
        draw_grid(im)
        plt.imshow(im[:, :, (2, 1, 0)])
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='g', facecolor="none")
        ax.add_patch(rect)
        plt.plot(cx, cy, 'ro', markersize=12)
        plt.title('Ground truth bounding box in YOLO format', fontsize=16)

        # Show YOLO mask
        ax = fig.add_subplot(1, 3, 3)
        plt.imshow(gt_mask)
        plt.title('Ground truth mask in YOLO format (7x7)', fontsize=16)
        plt.show()
