from matplotlib import pyplot as  plt
import torchvision.transforms as T
from skimage import morphology
from scipy import ndimage
from PIL import Image
import numpy as np
import pydicom
import csv



def transform_to_hu(medical_image, image):
    intercept = medical_image.RescaleIntercept
    slope = medical_image.RescaleSlope
    hu_image = image * slope + intercept
    return hu_image

def window_image(image, window_center, window_width):
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    window_image = image.copy()
    window_image[window_image < img_min] = img_min
    window_image[window_image > img_max] = img_max
    return window_image

def remove_noise(file_path, display=False):
    # https://vincentblog.xyz/posts/medical-images-in-python-computed-tomography
    medical_image = pydicom.read_file(file_path)
    image = medical_image.pixel_array
    hu_image = transform_to_hu(medical_image, image)
    brain_image = window_image(hu_image, 40, 80)

    segmentation = morphology.dilation(brain_image, np.ones((5, 5)))
    labels, label_nb = ndimage.label(segmentation)
    label_count = np.bincount(labels.ravel().astype(np.int))
    # The size of label_count is the number of classes/segmentations found
    # We don't use the first class since it's the background
    label_count[0] = 0
    # We create a mask with the class with more pixels
    # In this case should be the brain
    mask = labels == label_count.argmax()

    # Improve the brain mask
    mask = morphology.dilation(mask, np.ones((5, 5)))
    mask = ndimage.morphology.binary_fill_holes(mask)
    mask = morphology.dilation(mask, np.ones((3, 3)))
    # Since the the pixels in the mask are zero's and one's
    # We can multiple the original image to only keep the brain region
    masked_image = mask * brain_image

    if display:
        plt.figure(figsize=(15, 2.5))
        plt.subplot(141)
        plt.imshow(brain_image)
        plt.title('Original Image')
        plt.axis('off')
        plt.subplot(142)
        plt.imshow(mask)
        plt.title('Mask')
        plt.axis('off')
        plt.subplot(143)
        plt.imshow(masked_image)
        plt.title('Final Image')
        plt.axis('off')
    return masked_image


def crop_image(image, display=False):
    # Create a mask with the background pixels
    mask = image == 0
    # Find the brain area
    coords = np.array(np.nonzero(~mask))
    top_left = np.min(coords, axis=1)
    bottom_right = np.max(coords, axis=1)

    # Remove the background
    croped_image = image[top_left[0]:bottom_right[0],

                   top_left[1]:bottom_right[1]]

    return croped_image


def add_pad(image, new_height=512, new_width=512):
    height, width = image.shape
    final_image = np.zeros((new_height, new_width))
    pad_left = int((new_width - width) / 2)
    pad_top = int((new_height - height) / 2)

    # Replace the pixels with the image's pixels
    final_image[pad_top:pad_top + height, pad_left:pad_left + width] = image
    return final_image


def all_indices(value, qlist):
    indices = []
    idx = -1
    while True:
        try:
            idx = qlist.index(value, idx + 1)
            indices.append(idx)
        except ValueError:
            break
    return indices


def augment_data(image_orig, index_label=None):
    Hi = np.shape(image_orig)[0]
    Wi = np.shape(image_orig)[1]

    image = Image.fromarray(image_orig)
    transformHf = T.RandomHorizontalFlip(p=1)
    transform_20 = T.RandomRotation(degrees=(335, 345))
    transform20 = T.RandomRotation(degrees=(15, 25))
    image20 = transform20(image)
    image_20 = transform_20(image)
    imageHf = transformHf(image)
    imageHf20 = transform20(imageHf)
    imageHf_20 = transform_20(imageHf)
    return [np.array(image20).reshape(Hi, Wi), np.array(image_20).reshape(Hi, Wi),
            np.array(imageHf).reshape(Hi, Wi), np.array(imageHf20).reshape(Hi, Wi),
            np.array(imageHf_20).reshape(Hi, Wi)]



def read_lables(label_path):
    # hard codded fields as it is known based on the csv file
    Labels = {'scan': [], 'frame': [], 'Any': [], 'IPH': [], 'IVH': [], 'SAH': [], 'SDH': []}
    # with open('data_CTtraining/annotations.csv', newline='') as csvfile:
    with open(label_path, newline='') as csvfile:
        labelReader = csv.reader(csvfile, delimiter=',', quotechar='|')
        next(labelReader)
        scan_counter = 0
        for row in labelReader:
            Labels['scan'].append(row[1])
            Labels['frame'].append(row[0])
            Labels['Any'].append(int(row[2]))
            Labels['IPH'].append(int(row[3]))
            Labels['IVH'].append(int(row[4]))
            Labels['SAH'].append(int(row[5]))
            Labels['SDH'].append(int(row[6]))
    return Labels


def write_lables(label_path, append='a', cols=None):
    fieldnames = ['slice_id', 'case_id', 'ANY', 'IPH', 'IVH', 'SAH', 'SDH']
    with open(label_path, append, newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if cols:
            writer.writerow({fieldnames[0]: cols[0], fieldnames[1]: cols[1],
                             fieldnames[2]: cols[2], fieldnames[3]: cols[3], fieldnames[4]: cols[4],
                             fieldnames[5]: cols[5], fieldnames[6]: cols[6]})
        else:
            writer.writeheader()

