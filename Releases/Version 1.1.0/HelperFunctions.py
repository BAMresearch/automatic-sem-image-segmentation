import os
import math
import numpy as np
import random
from shutil import copy
from skimage.filters import threshold_otsu, threshold_li
import cv2
from PIL import Image
from tqdm import tqdm

import Measurements
from Measurements import Measure


def tile_image(img, tile_size_w, tile_size_h, min_overlap=2, normalization_range=None, normalize_tiles_individually=True):
    image_size_w = img.shape[1]
    image_size_h = img.shape[0]

    no_of_x_tiles = math.ceil(image_size_w / tile_size_w)
    no_of_y_tiles = math.ceil(image_size_h / tile_size_h)

    # If more than 1 tile has to be used introduce at least 'minOverlap' pixel overlap between tiles to avoid edge seams
    if (no_of_x_tiles > 1) and ((tile_size_w - (image_size_w % tile_size_w)) % tile_size_w <= min_overlap):
        no_of_x_tiles += 1
    if (no_of_y_tiles > 1) and ((tile_size_h - (image_size_h % tile_size_h)) % tile_size_h <= min_overlap):
        no_of_y_tiles += 1
    no_of_tiles = no_of_x_tiles * no_of_y_tiles

    img_tiles = np.zeros((no_of_tiles, tile_size_h, tile_size_w, 1), dtype='float32')

    # Tile image if it is bigger than the input tensor (use overlapping tiles), convert to float, normalize to [0, 1]
    k = 0
    for i in range(0, no_of_x_tiles):
        if no_of_x_tiles > 1:
            offset_x = math.ceil(i * (tile_size_w - ((tile_size_w * no_of_x_tiles - image_size_w) / (no_of_x_tiles - 1))))
        else:
            offset_x = 0

        for j in range(0, no_of_y_tiles):
            if no_of_y_tiles > 1:
                offset_y = math.ceil(j * (tile_size_h - ((tile_size_h * no_of_y_tiles - image_size_h) / (no_of_y_tiles - 1))))
            else:
                offset_y = 0

            img_tiles[k, :, :, :] = img[offset_y:min(offset_y + tile_size_h, image_size_h), offset_x:min(offset_x + tile_size_w, image_size_w), :]

            k += 1

    if normalization_range is not None:
        if normalize_tiles_individually:
            for i in range(0, img_tiles.shape[0]):
                img_tiles[i] -= np.min(img_tiles[i])
                img_tiles[i] /= np.max(img_tiles[i])
                img_tiles[i] = normalization_range[0] + (normalization_range[1] - normalization_range[0]) * img_tiles[i]
        else:
            img_tiles -= np.min(img)
            img_tiles /= np.max(img)
            img_tiles = normalization_range[0] + (normalization_range[1] - normalization_range[0]) * img_tiles

    return img_tiles


def stitch_image(img, image_size_w, image_size_h, min_overlap=2, manage_overlap_mode=2, return_8_bit_image=False):
    input_size_w = img.shape[2]
    input_size_h = img.shape[1]

    no_of_x_tiles = math.ceil(image_size_w / input_size_w)
    no_of_y_tiles = math.ceil(image_size_h / input_size_h)

    # If more than 1 tile has to be used introduce at least 'minOverlap' pixel overlap between tiles to avoid edge seams
    if (no_of_x_tiles > 1) and ((input_size_w - (image_size_w % input_size_w)) % input_size_w <= min_overlap):
        no_of_x_tiles += 1
    if (no_of_y_tiles > 1) and ((input_size_h - (image_size_h % input_size_h)) % input_size_h <= min_overlap):
        no_of_y_tiles += 1

    img_stitched = np.zeros((image_size_h, image_size_w, img.shape[-1]), dtype='float32')
    overlaps = np.zeros_like(img_stitched, dtype='uint8')

    k = 0
    overlap_size_x = 0
    overlap_size_y = 0
    if no_of_x_tiles > 1:
        overlap_size_x = (input_size_w * no_of_x_tiles - image_size_w) // (2 * (no_of_x_tiles - 1))
    if no_of_y_tiles > 1:
        overlap_size_y = (input_size_h * no_of_y_tiles - image_size_h) // (2 * (no_of_y_tiles - 1))

    for i in range(0, no_of_x_tiles):
        if no_of_x_tiles > 1:
            offset_x = math.ceil(i * (input_size_w - ((input_size_w * no_of_x_tiles - image_size_w) / (no_of_x_tiles - 1))))
        else:
            offset_x = 0

        for j in range(0, no_of_y_tiles):
            if no_of_y_tiles > 1:
                offset_y = math.ceil(
                    j * (input_size_h - ((input_size_h * no_of_y_tiles - image_size_h) / (no_of_y_tiles - 1))))
            else:
                offset_y = 0

            if manage_overlap_mode == 0:
                # Take maximum in overlapping regions
                img_stitched[offset_y:min(offset_y + input_size_h, image_size_h), offset_x:min(offset_x + input_size_w, image_size_w), :] = np.maximum(img[k, :, :, :], img_stitched[offset_y:min(offset_y + input_size_h, image_size_h), offset_x:min(offset_x + input_size_w, image_size_w), :])
            elif manage_overlap_mode == 1:
                # Average in overlapping regions
                img_stitched[offset_y:min(offset_y + input_size_h, image_size_h), offset_x:min(offset_x + input_size_w, image_size_w), :] += img[k, :, :, :]
                overlaps[offset_y:min(offset_y + input_size_h, image_size_h), offset_x:min(offset_x + input_size_w, image_size_w), :] += np.ones((input_size_h, input_size_w, img.shape[-1]), dtype='uint8')
            elif manage_overlap_mode == 2:
                # Crop overlapping regions
                if i == 0:
                    cxl = 0 * overlap_size_x  # left
                    cxr = 1 * overlap_size_x  # right
                elif i == no_of_x_tiles - 1:
                    cxl = 1 * overlap_size_x
                    cxr = 0 * overlap_size_x
                else:
                    cxl = 1 * overlap_size_x
                    cxr = 1 * overlap_size_x
                if j == 0:
                    cyt = 0 * overlap_size_y  # top
                    cyb = 1 * overlap_size_y  # bottom
                elif j == no_of_y_tiles - 1:
                    cyt = 1 * overlap_size_y
                    cyb = 0 * overlap_size_y
                else:
                    cyt = 1 * overlap_size_y
                    cyb = 1 * overlap_size_y
                img_stitched[offset_y + cyt:min(offset_y + input_size_h - cyb, image_size_h), offset_x + cxl:min(offset_x + input_size_w - cxr, image_size_w), :] = img[k, cyt:input_size_h - cyb, cxl:input_size_w - cxr, :]

            k += 1

    if manage_overlap_mode == 1:  # Average
        img_stitched = np.asarray(img_stitched / overlaps, dtype='float32')
    else:
        img_stitched = np.asarray(img_stitched, dtype='float32')

    if return_8_bit_image:
        img_stitched = np.asarray(img_stitched * 255, dtype='uint8')

    return img_stitched


def eight_to_four_connected(img):
    if np.count_nonzero(img) > 2 or np.count_nonzero(img) < img.size - 2:  # If there are less than two 0 or 1 entries in the entire image just use the image as is
        for x in range(0, img.shape[0]-1):
            for y in range(0, img.shape[1]-1):
                if img[x, y] == 0 and img[x + 1, y + 1] == 0 and img[x + 1, y] != 0 and img[x, y + 1] != 0:
                    img[x + 1, y] = 0
                elif img[x + 1, y] == 0 and img[x, y + 1] == 0 and img[x, y] != 0 and img[x + 1, y + 1] != 0:
                    img[x, y] = 0
    return img


def segment(image, threshold, watershed_lines, min_distance=9, use_four_connectivity=True):
    labels = Measurements.Measure.segment(image, threshold, watershed_lines, min_distance, darkBackground=True)
    if use_four_connectivity:
        labels = eight_to_four_connected(labels)

    return labels


def filter_gan_masks(img_path, msk_path, out_path, threshold_method=threshold_li, do_watershed_and_four_connectivity=True):
    for f in tqdm(os.listdir(img_path)):
        img = np.asarray(Image.open(os.path.join(img_path, f)), dtype='uint8')
        mask = np.asarray(Image.open(os.path.join(msk_path, f)), dtype='uint8')
        if do_watershed_and_four_connectivity:
            mask = segment(image=mask, threshold=-1, watershed_lines=True, use_four_connectivity=True)
        m = Measure(mask, darkBackground=True, applyWatershed=False, excludeEdges=False, grayscaleImage=img)
        m.calculateMeanIntensities()
        m.filterResults('meanIntensity', threshold_method(img))

        contours = np.zeros(img.shape, dtype='uint8')
        cv2.drawContours(image=contours, contours=m.contours, contourIdx=-1, color=(255, 255, 255), thickness=-1)

        Image.fromarray(contours).save(os.path.join(out_path, f))


def initialize_directories(root_dir, output_dir_cyclegan, output_dir_unet):
    # WGAN
    wgan_dir = os.path.join(root_dir, '1_WGAN')
    if not os.path.isdir(wgan_dir):
        os.mkdir(wgan_dir)
    if not os.path.isdir(os.path.join(wgan_dir, 'Output_Images')):
        os.mkdir(os.path.join(wgan_dir, 'Output_Images'))
    if not os.path.isdir(os.path.join(wgan_dir, 'Models')):
        os.mkdir(os.path.join(wgan_dir, 'Models'))

    # CycleGAN
    cycle_gan_dir = os.path.join(root_dir, '2_CycleGAN')
    if not os.path.isdir(cycle_gan_dir):
        os.mkdir(cycle_gan_dir)
    if not os.path.isdir(os.path.join(cycle_gan_dir, 'data')):
        os.mkdir(os.path.join(cycle_gan_dir, 'data'))
    if not os.path.isdir(os.path.join(cycle_gan_dir, 'generate_images')):
        os.mkdir(os.path.join(cycle_gan_dir, 'generate_images'))
    if not os.path.isdir(os.path.join(cycle_gan_dir, 'images')):
        os.mkdir(os.path.join(cycle_gan_dir, 'images'))
    if not os.path.isdir(os.path.join(cycle_gan_dir, 'Models')):
        os.mkdir(os.path.join(cycle_gan_dir, 'Models'))

    if not os.path.isdir(os.path.join(cycle_gan_dir, 'data', 'testA')):
        os.mkdir(os.path.join(cycle_gan_dir, 'data', 'testA'))
    if not os.path.isdir(os.path.join(cycle_gan_dir, 'data', 'testB')):
        os.mkdir(os.path.join(cycle_gan_dir, 'data', 'testB'))
    if not os.path.isdir(os.path.join(cycle_gan_dir, 'data', 'trainA')):
        os.mkdir(os.path.join(cycle_gan_dir, 'data', 'trainA'))
    if not os.path.isdir(os.path.join(cycle_gan_dir, 'data', 'trainB')):
        os.mkdir(os.path.join(cycle_gan_dir, 'data', 'trainB'))

    if not os.path.isdir(os.path.join(cycle_gan_dir, 'generate_images', 'A')):
        os.mkdir(os.path.join(cycle_gan_dir, 'generate_images', 'A'))
    if not os.path.isdir(os.path.join(cycle_gan_dir, 'generate_images', 'B')):
        os.mkdir(os.path.join(cycle_gan_dir, 'generate_images', 'B'))
    if not os.path.isdir(os.path.join(cycle_gan_dir, 'generate_images', 'Synthetic_Masks_Filtered')):
        os.mkdir(os.path.join(cycle_gan_dir, 'generate_images', 'Synthetic_Masks_Filtered'))

    if not os.path.isdir(output_dir_cyclegan):
        os.mkdir(output_dir_cyclegan)

    # UNet
    u_net_dir = os.path.join(root_dir, '3_UNet')
    if not os.path.isdir(u_net_dir):
        os.mkdir(u_net_dir)
    if not os.path.isdir(os.path.join(u_net_dir, 'Models')):
        os.mkdir(os.path.join(u_net_dir, 'Models'))

    if not os.path.isdir(output_dir_unet):
        os.mkdir(output_dir_unet)


def prepare_images_cycle_gan(root_dir, input_dir_images, tile_size_w=384, tile_size_h=384, num_simulated_masks=1000):
    # Tile SEM Images to correct Size
    input_imgs = load_and_preprocess_images(input_dir_or_filelist=input_dir_images, normalization_range=None, output_channels=1)
    filenames = get_image_file_paths_from_directory(input_dir_images)
    for i, input_img in enumerate(input_imgs):
        img_tiles = np.asarray(tile_image(input_img, tile_size_w, tile_size_h, normalization_range=None, min_overlap=0), dtype='uint8')
        f = os.path.split(filenames[i])[-1]
        for j, img_tile in enumerate(img_tiles):
            # Filter out tiles that show mainly background for training
            if np.mean(img_tile) >= 1.1 * np.mean(input_img):
                ext = os.path.splitext(f)[-1]
                Image.fromarray(img_tile[:, :, 0]).save(os.path.join(root_dir, '2_CycleGAN', 'data', 'trainA', f.replace(ext, f'-{j}{ext}')))

    # Choose 5 random files for testing
    filenames = get_image_file_paths_from_directory(os.path.join(root_dir, '2_CycleGAN', 'data', 'trainA'))
    test_img = random.sample(filenames, 5)
    input_dir = os.path.join(root_dir, '2_CycleGAN', 'data', 'trainA')
    output_dir = os.path.join(root_dir, '2_CycleGAN', 'data', 'testA')
    for f in test_img:
        copy(os.path.join(input_dir, f), output_dir)

    # Make sure there are the same number of images from each domain
    num_images_a = len([f for f in os.listdir(os.path.join(root_dir, '2_CycleGAN', 'data', 'trainA'))])
    i = 0

    # If there are less image tiles than the requested number of simulated masks, use random crops/augmentations to match the number of masks
    if num_images_a < num_simulated_masks:
        print('Performing data augmentation...')
        with tqdm(total=num_simulated_masks - num_images_a) as pbar:
            while i < num_simulated_masks - num_images_a:
                r = random.randint(0, input_imgs.shape[0] - 1)
                f = os.path.split(filenames[r])[-1]
                ext = os.path.splitext(f)[-1]
                input_img = input_imgs[r, :, :, :]
                a = random.randint(0, input_img.shape[0] - tile_size_h - 1)
                b = random.randint(0, input_img.shape[1] - tile_size_w - 1)
                img_tile = input_img[a:a + tile_size_h, b:b + tile_size_w]
                if random.random() > 0.5:
                    img_tile = np.fliplr(img_tile)
                if random.random() > 0.5:
                    img_tile = np.flipud(img_tile)

                # Filter out tiles that show mainly background for training
                if np.mean(img_tile) >= 1.1 * np.mean(input_img):
                    Image.fromarray(img_tile[:, :, 0].astype('uint8')).save(os.path.join(root_dir, '2_CycleGAN', 'data', 'trainA', f.replace(ext, f'-aug_{i}{ext}')))
                    i += 1
                    pbar.update(1)


def get_image_file_paths_from_directory(directory):
    return [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.tif') or file.endswith('.tiff') or file.endswith('.png') or file.endswith('.bmp') or file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.gif')]


def load_and_preprocess_images(input_dir_or_filelist, threshold_value=None, normalization_range=(-1, 1), output_channels=1, contrast_optimization_range=None):
    images = []
    if isinstance(input_dir_or_filelist, str) or isinstance(input_dir_or_filelist, os.PathLike):
        if os.path.isdir(input_dir_or_filelist):
            file_list = get_image_file_paths_from_directory(input_dir_or_filelist)
        else:
            file_list = [input_dir_or_filelist, ]
    else:
        file_list = input_dir_or_filelist

    for file in file_list:
        image = np.asarray(Image.open(file), dtype='float32')

        assert len(image.shape) >= 2 and len(image.shape) <= 3 and output_channels in (1, 3), 'Invalid Image format'

        if len(image.shape) == 3 and output_channels == 1:  # convert to grayscale (simply average here)
            image = np.average(image, -1)
        elif len(image.shape) == 2:
            image = image[:, :, np.newaxis]

        if contrast_optimization_range is not None and contrast_optimization_range[0] > 0 and contrast_optimization_range[1] < 100:
            lb = np.percentile(image, contrast_optimization_range[0])
            ub = np.percentile(image, contrast_optimization_range[1])
            image = np.where(image <= lb, lb, image)
            image = np.where(image >= ub, ub, image)

        if normalization_range is not None:
            image -= np.min(image)
            image /= np.max(image)
            if threshold_value is not None:
                image = image > threshold_value
            image = normalization_range[0] + (normalization_range[1] - normalization_range[0]) * image

        images.append(image)

    return np.asarray(images, dtype='float32')
