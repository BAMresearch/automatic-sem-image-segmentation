import os
import math
import numpy as np
import imageio
import random
from shutil import copy
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from scipy import ndimage
from skimage.filters import threshold_otsu, threshold_li
import cv2

from Measurements import Measure
from datetime import datetime

ROOT_DIR = os.path.abspath("./")
INPUT_DIR_MASKS = os.path.join(ROOT_DIR, 'Input_Masks')
INPUT_DIR_IMAGES = os.path.join(ROOT_DIR, 'Input_Images')
OUTPUT_DIR_CYCLEGAN = os.path.join(ROOT_DIR, 'Output_Masks_CycleGAN')
OUTPUT_DIR_UNET = os.path.join(ROOT_DIR, 'Output_Masks_UNet')
IMAGE_SIZE_W = 1024
IMAGE_SIZE_H = 712
TILE_SIZE_W = 512
TILE_SIZE_H = 352

def tileImage(img, IMAGE_SIZE_W, IMAGE_SIZE_H, INPUT_SIZE_W, INPUT_SIZE_H, minOverlap=2, normalizeOutput=True):
    noOfXTiles = math.ceil(IMAGE_SIZE_W/INPUT_SIZE_W)
    noOfYTiles = math.ceil(IMAGE_SIZE_H/INPUT_SIZE_H)

    # If more than 1 tile has to be used introduce at least 'minOverlap' pixel overlap between tiles to avoid edge seams
    if ((noOfXTiles > 1) and ((INPUT_SIZE_W - (IMAGE_SIZE_W % INPUT_SIZE_W)) % INPUT_SIZE_W <= minOverlap)):
        noOfXTiles += 1
    if ((noOfYTiles > 1) and ((INPUT_SIZE_H - (IMAGE_SIZE_H % INPUT_SIZE_H)) % INPUT_SIZE_H <= minOverlap)):
        noOfYTiles += 1
    noOfTiles = noOfXTiles * noOfYTiles

    imgTiles = np.zeros((noOfTiles, INPUT_SIZE_H, INPUT_SIZE_W, 1), dtype='float32')

    # Tile image if it is bigger than the input tensor (use overlapping tiles), convert to float, normalize to [0, 1]
    k = 0
    offsetX = 0
    offsetY = 0
    for i in range(0, noOfXTiles):
        if (noOfXTiles > 1):
            offsetX = math.ceil(i * (INPUT_SIZE_W - ((INPUT_SIZE_W * noOfXTiles - IMAGE_SIZE_W) / (noOfXTiles-1))))
        else:
            offsetX = 0

        for j in range(0, noOfYTiles):
            if (noOfYTiles > 1):
                offsetY = math.ceil(j * (INPUT_SIZE_H - ((INPUT_SIZE_H * noOfYTiles - IMAGE_SIZE_H) / (noOfYTiles-1))))
            else:
                offsetY = 0

            imgTiles[k, :, :, 0] = img[offsetY:min(offsetY+INPUT_SIZE_H, IMAGE_SIZE_H), offsetX:min(offsetX+INPUT_SIZE_W, IMAGE_SIZE_W)]

            k += 1

    if normalizeOutput:
        imgTiles -= np.min(img)
        imgTiles /= np.max(img)
    return imgTiles


def stitchImage(img, IMAGE_SIZE_W, IMAGE_SIZE_H, INPUT_SIZE_W, INPUT_SIZE_H, minOverlap=2, manageOverlapsMode=2, return8BitImage=False):
    noOfXTiles = math.ceil(IMAGE_SIZE_W / INPUT_SIZE_W)
    noOfYTiles = math.ceil(IMAGE_SIZE_H / INPUT_SIZE_H)

    # If more than 1 tile has to be used introduce at least 'minOverlap' pixel overlap between tiles to avoid edge seams
    if ((noOfXTiles > 1) and ((INPUT_SIZE_W - (IMAGE_SIZE_W % INPUT_SIZE_W)) % INPUT_SIZE_W <= minOverlap)):
        noOfXTiles += 1
    if ((noOfYTiles > 1) and ((INPUT_SIZE_H - (IMAGE_SIZE_H % INPUT_SIZE_H)) % INPUT_SIZE_H <= minOverlap)):
        noOfYTiles += 1
    noOfTiles = noOfXTiles * noOfYTiles

    imgStitched = np.zeros((IMAGE_SIZE_H, IMAGE_SIZE_W, img.shape[-1]), dtype='float32')
    overlaps = np.zeros_like(imgStitched, dtype='uint8')

    k = 0
    offsetX = 0
    offsetY = 0
    overlapSizeX = (INPUT_SIZE_W * noOfXTiles - IMAGE_SIZE_W) // (2 * (noOfXTiles - 1))
    overlapSizeY = (INPUT_SIZE_H * noOfYTiles - IMAGE_SIZE_H) // (2 * (noOfYTiles - 1))
    for i in range(0, noOfXTiles):
        if (noOfXTiles > 1):
            offsetX = math.ceil(i * (INPUT_SIZE_W - ((INPUT_SIZE_W * noOfXTiles - IMAGE_SIZE_W) / (noOfXTiles - 1))))
        else:
            offsetX = 0

        for j in range(0, noOfYTiles):
            if (noOfYTiles > 1):
                offsetY = math.ceil(
                    j * (INPUT_SIZE_H - ((INPUT_SIZE_H * noOfYTiles - IMAGE_SIZE_H) / (noOfYTiles - 1))))
            else:
                offsetY = 0

            if manageOverlapsMode == 0:
                # Take maximum in overlapping regions
                imgStitched[offsetY:min(offsetY + INPUT_SIZE_H, IMAGE_SIZE_H), offsetX:min(offsetX + INPUT_SIZE_W, IMAGE_SIZE_W), :] = np.maximum(img[k, :, :, :], imgStitched[offsetY:min(offsetY + INPUT_SIZE_H, IMAGE_SIZE_H), offsetX:min(offsetX + INPUT_SIZE_W, IMAGE_SIZE_W), :])
            elif manageOverlapsMode == 1:
                # Average in overlapping regions
                imgStitched[offsetY:min(offsetY + INPUT_SIZE_H, IMAGE_SIZE_H), offsetX:min(offsetX + INPUT_SIZE_W, IMAGE_SIZE_W), :] += img[k, :, :, :]
                overlaps[offsetY:min(offsetY + INPUT_SIZE_H, IMAGE_SIZE_H), offsetX:min(offsetX + INPUT_SIZE_W, IMAGE_SIZE_W), :] += np.ones((INPUT_SIZE_H, INPUT_SIZE_W, img.shape[-1]), dtype='uint8')
            elif manageOverlapsMode == 2:
                # Crop overlapping regions
                if i == 0:
                    cxl = 0 * overlapSizeX  # left
                    cxr = 1 * overlapSizeX  # right
                elif i == noOfXTiles - 1:
                    cxl = 1 * overlapSizeX
                    cxr = 0 * overlapSizeX
                else:
                    cxl = 1 * overlapSizeX
                    cxr = 1 * overlapSizeX
                if j == 0:
                    cyt = 0 * overlapSizeY  # top
                    cyb = 1 * overlapSizeY  # bottom
                elif j == noOfYTiles - 1:
                    cyt = 1 * overlapSizeY
                    cyb = 0 * overlapSizeY
                else:
                    cyt = 1 * overlapSizeY
                    cyb = 1 * overlapSizeY
                imgStitched[offsetY + cyt:min(offsetY + INPUT_SIZE_H - cyb, IMAGE_SIZE_H), offsetX + cxl:min(offsetX + INPUT_SIZE_W - cxr, IMAGE_SIZE_W), :] = img[k, cyt:INPUT_SIZE_H - cyb, cxl:INPUT_SIZE_W - cxr, :]

            k += 1

    if manageOverlapsMode == 1:  # Average
        imgStitched = np.asarray(imgStitched / overlaps, dtype='float32')
    else:
        imgStitched = np.asarray(imgStitched, dtype='float32')

    if return8BitImage:
        imgStitched = np.asarray(imgStitched * 255, dtype='uint8')

    return imgStitched


def eightToFourConnected(img):
    if np.count_nonzero(img) > 2 or np.count_nonzero(img) < img.size - 2: # If there are less than two 0 or 1 entries in the entire image just use the image as is
        for x in range(0, img.shape[0]-1):
            for y in range(0, img.shape[1]-1):
                if img[x, y] == 0 and img[x + 1, y + 1] == 0 and img[x + 1, y] != 0 and img[x, y + 1] != 0:
                    img[x + 1, y] = 0
                elif img[x + 1, y] == 0 and img[x, y + 1] == 0 and img[x, y] != 0 and img[x + 1, y + 1] != 0:
                    img[x, y] = 0
    return img


def segment(image, threshold, watershedLines):
    MIN_DISTANCE = 9

    img = image.copy()
    # Threshold the image if it is not already a bool
    if not img.dtype == np.bool:
        if threshold < 0:
            threshold = threshold_otsu(img)
        mask = (img > threshold)
    else:
        mask = img

    # If all values are the same after thresholding, skip watershed and return image directly
    if np.min(mask) == np.max(mask):
        return np.asarray(mask > 0, dtype='uint8')

    # Get the Euclidian Distance Map and smooth it
    distance = ndimage.distance_transform_edt(mask)
    distance = ndimage.gaussian_filter(distance, sigma=1)

    # Get the local maxima of the distance map
    local_maxi = peak_local_max(distance, indices=False, min_distance=MIN_DISTANCE)

    # Merge maxima that are close by
    # local_maxi = ndimage.binary_dilation(local_maxi, iterations=2)

    markers = ndimage.label(local_maxi)[0]
    labels = watershed(-distance, markers, mask=mask, watershed_line=watershedLines)
    labels = labels > 0
    labels = ndimage.binary_fill_holes(labels, structure=np.ones((3, 3)))

    return np.asarray(labels * 255, dtype='uint8')


def filterGANMasks(imgPath, mskPath, outPath, doWatershedAndFourConnectivity=True):
    for f in os.listdir(imgPath):
        img = np.asarray(imageio.imread(os.path.join(imgPath, f)), dtype='uint8').copy()
        mask = np.asarray(imageio.imread(os.path.join(mskPath, f).replace('_synthetic.tif', '.tif')), dtype='uint8').copy()
        if doWatershedAndFourConnectivity:
            mask = segment(mask, -1, True)
            mask = eightToFourConnected(mask)
        m = Measure(mask, darkBackground=True, applyWatershed=False, excludeEdges=False, grayscaleImage=img)
        m.calculateMeanIntensities()
        # m.filterResults('meanIntensity', np.median(img) + np.std(img))
        m.filterResults('meanIntensity', threshold_li(img))

        contours = np.zeros(img.shape, dtype='uint8')
        cv2.drawContours(image=contours, contours=m.contours, contourIdx=-1, color=(255, 255, 255), thickness=-1)

        imageio.imwrite(os.path.join(outPath, f).replace('_synthetic.tif', '.tif'), contours)

def initializeDirectories():
    # WGAN
    wGANDir = os.path.join(ROOT_DIR, '1_WGAN')
    if not os.path.isdir(wGANDir):
        os.mkdir(wGANDir)
    if not os.path.isdir(os.path.join(wGANDir, 'Output_Images')):
        os.mkdir(os.path.join(wGANDir, 'Output_Images'))
    if not os.path.isdir(os.path.join(wGANDir, 'Models')):
        os.mkdir(os.path.join(wGANDir, 'Models'))
    if not os.path.isdir(os.path.join(wGANDir, 'Generated_Masks')):
        os.mkdir(os.path.join(wGANDir, 'Generated_Masks'))

    # CycleGAN
    cycleGANDir = os.path.join(ROOT_DIR, '2_CycleGAN')
    if not os.path.isdir(cycleGANDir):
        os.mkdir(cycleGANDir)
    if not os.path.isdir(os.path.join(cycleGANDir, 'data')):
        os.mkdir(os.path.join(cycleGANDir, 'data'))
    if not os.path.isdir(os.path.join(cycleGANDir, 'generate_images')):
        os.mkdir(os.path.join(cycleGANDir, 'generate_images'))
    if not os.path.isdir(os.path.join(cycleGANDir, 'images')):
        os.mkdir(os.path.join(cycleGANDir, 'images'))
    if not os.path.isdir(os.path.join(cycleGANDir, 'saved_models')):
        os.mkdir(os.path.join(cycleGANDir, 'saved_models'))

    if not os.path.isdir(os.path.join(cycleGANDir, 'data', 'testA')):
        os.mkdir(os.path.join(cycleGANDir, 'data', 'testA'))
    if not os.path.isdir(os.path.join(cycleGANDir, 'data', 'testB')):
        os.mkdir(os.path.join(cycleGANDir, 'data', 'testB'))
    if not os.path.isdir(os.path.join(cycleGANDir, 'data', 'trainA')):
        os.mkdir(os.path.join(cycleGANDir, 'data', 'trainA'))
    if not os.path.isdir(os.path.join(cycleGANDir, 'data', 'trainB')):
        os.mkdir(os.path.join(cycleGANDir, 'data', 'trainB'))

    if not os.path.isdir(os.path.join(cycleGANDir, 'generate_images', 'models')):
        os.mkdir(os.path.join(cycleGANDir, 'generate_images', 'models'))
    if not os.path.isdir(os.path.join(cycleGANDir, 'generate_images', 'synthetic_images')):
        os.mkdir(os.path.join(cycleGANDir, 'generate_images', 'synthetic_images'))
    if not os.path.isdir(os.path.join(cycleGANDir, 'generate_images', 'synthetic_images', 'A')):
        os.mkdir(os.path.join(cycleGANDir, 'generate_images', 'synthetic_images', 'A'))
    if not os.path.isdir(os.path.join(cycleGANDir, 'generate_images', 'synthetic_images', 'B')):
        os.mkdir(os.path.join(cycleGANDir, 'generate_images', 'synthetic_images', 'B'))
    if not os.path.isdir(os.path.join(cycleGANDir, 'generate_images', 'synthetic_images', 'trainA')):
        os.mkdir(os.path.join(cycleGANDir, 'generate_images', 'synthetic_images', 'trainA'))
    if not os.path.isdir(os.path.join(cycleGANDir, 'generate_images', 'synthetic_images', 'trainB')):
        os.mkdir(os.path.join(cycleGANDir, 'generate_images', 'synthetic_images', 'trainB'))
    if not os.path.isdir(os.path.join(cycleGANDir, 'generate_images', 'synthetic_images', 'testA')):
        os.mkdir(os.path.join(cycleGANDir, 'generate_images', 'synthetic_images', 'testA'))
    if not os.path.isdir(os.path.join(cycleGANDir, 'generate_images', 'synthetic_images', 'testB')):
        os.mkdir(os.path.join(cycleGANDir, 'generate_images', 'synthetic_images', 'testB'))
    if not os.path.isdir(os.path.join(cycleGANDir, 'generate_images', 'synthetic_images', 'testB_Filtered')):
        os.mkdir(os.path.join(cycleGANDir, 'generate_images', 'synthetic_images', 'testB_Filtered'))
    if not os.path.isdir(os.path.join(cycleGANDir, 'generate_images', 'synthetic_images', 'B_stitched_unfiltered')):
        os.mkdir(os.path.join(cycleGANDir, 'generate_images', 'synthetic_images', 'B_stitched_unfiltered'))

    if not os.path.isdir(OUTPUT_DIR_CYCLEGAN):
        os.mkdir(OUTPUT_DIR_CYCLEGAN)

    # UNet
    uNETDir = os.path.join(ROOT_DIR, '3_UNet')
    if not os.path.isdir(uNETDir):
        os.mkdir(uNETDir)
    if not os.path.isdir(os.path.join(uNETDir, 'Models')):
        os.mkdir(os.path.join(uNETDir, 'Models'))
    if not os.path.isdir(os.path.join(uNETDir, 'Input')):
        os.mkdir(os.path.join(uNETDir, 'Input'))
    if not os.path.isdir(os.path.join(uNETDir, 'Output')):
        os.mkdir(os.path.join(uNETDir, 'Output'))

    if not os.path.isdir(OUTPUT_DIR_UNET):
        os.mkdir(OUTPUT_DIR_UNET)

def prepareImagesCycleGAN():
    # Tile SEM Images to correct Size
    for f in os.listdir(INPUT_DIR_IMAGES):
        inputImg = np.asarray(imageio.imread(os.path.join(INPUT_DIR_IMAGES, f)), dtype='float32').copy()
        imgTiles = np.asarray(tileImage(inputImg, IMAGE_SIZE_W, IMAGE_SIZE_H, TILE_SIZE_W, TILE_SIZE_H, normalizeOutput=False), dtype='uint8')
        for i, imgTile in enumerate(imgTiles):
            # Save all tiles for later inference
            imageio.imwrite(os.path.join(ROOT_DIR, '2_CycleGAN', 'generate_images', 'synthetic_images', 'testA', f.replace('.tif', '-' + str(i) + '.tif')), imgTile[:, :, 0])
            # Filter out tiles that show mainly background for training
            if np.mean(imgTile) >= 1.1 * np.mean(inputImg):
                imageio.imwrite(os.path.join(ROOT_DIR, '2_CycleGAN', 'data', 'trainA', f.replace('.tif', '-' + str(i) + '.tif')), imgTile[:, :, 0])

    # Copy Files to their directories for training and inference
    inputDir = os.path.join(ROOT_DIR, '1_WGAN', 'Generated_Masks')
    outputDir = os.path.join(ROOT_DIR, '2_CycleGAN', 'generate_images', 'synthetic_images', 'testB')
    for f in os.listdir(inputDir):
        copy(os.path.join(inputDir, f), outputDir)

    inputFiles = [f for f in os.listdir(os.path.join(ROOT_DIR, '2_CycleGAN', 'data', 'trainA'))]
    # Choose 10 random files for testing
    testImg = random.sample(inputFiles, 10)
    inputDir = os.path.join(ROOT_DIR, '2_CycleGAN', 'data', 'trainA')
    outputDir = os.path.join(ROOT_DIR, '2_CycleGAN', 'data', 'testA')
    for f in testImg:
        copy(os.path.join(inputDir, f), outputDir)

    # Choose some random Masks (same number as SEM image tiles, or all available tiles if less) for training and testing
    inputDir = os.path.join(ROOT_DIR, '2_CycleGAN', 'generate_images', 'synthetic_images', 'testB')
    fileList = [f for f in os.listdir(inputDir)]
    testImg = random.sample(fileList, min(len(inputFiles) + 10, len(fileList)))
    for i, f in enumerate(testImg):
        if i < 10:
            outputDir = os.path.join(ROOT_DIR, '2_CycleGAN', 'data', 'testB')
            copy(os.path.join(inputDir, f), outputDir)
        else:
            outputDir = os.path.join(ROOT_DIR, '2_CycleGAN', 'data', 'trainB')
            copy(os.path.join(inputDir, f), outputDir)


if __name__ == '__main__':
    initializeDirectories()
    # Step 1: Train WGAN
    # WGAN = WassersteinGAN.WGAN(ROOT_DIR=ROOT_DIR, instance_image_shape=(64, 64, 1), image_shape=(352, 512, 1))
    # trainedWGANModel = WGAN.startTraining()
    #
    # Step 2: Simulate Masks
    # WGAN.simulateMasks(trainedWGANModel, NO_OF_IMAGES=250, USE_PERLIN_NOISE=True)

    # Step 1&2: Train WGAN and simulate Masks
    # Workaround for tensorflow not freeing GPU memory on RTX cards after process completion: start in own interpreter
    os.system('python WassersteinGAN.py {} --instance_image_shape=(64,64,1) --image_shape=({},{},1) --image_number=1000'.format(ROOT_DIR, TILE_SIZE_H, TILE_SIZE_W))
    # Step 3: Train cycleGAN
    prepareImagesCycleGAN()
    # cycleGAN = CycleGAN.CycleGAN(image_shape=(352, 512, 1), ROOT_DIR=ROOT_DIR)
    # cycleGAN.runTraining()
    # Workaround for tensorflow not freeing GPU memory on RTX cards after process completion: start in own interpreter

    os.system('python CycleGAN.py {} --image_shape=({},{},1) --mode=training'.format(ROOT_DIR, TILE_SIZE_H, TILE_SIZE_W))

    # Step 4: Simulate fake SEM Images and segment real SEM images with cycleGAN
    # cycleGAN.runInference()
    # Workaround for tensorflow not freeing GPU memory on RTX cards after process completion: start in own interpreter
    os.system('python CycleGAN.py {} --image_shape=({},{},1) --mode=inference'.format(ROOT_DIR, TILE_SIZE_H, TILE_SIZE_W))

    # Stitch image tiles back together
    for f in os.listdir(INPUT_DIR_IMAGES):
        i = 0
        imgTiles = []
        g = os.path.join(ROOT_DIR, '2_CycleGAN', 'generate_images', 'synthetic_images', 'B', f.replace('.tif', '-' + str(i) + '_synthetic.tif'))
        while os.path.exists(g):
            imgTiles.append(np.asarray(imageio.imread(g), dtype='float32').copy())
            i += 1
            g = os.path.join(ROOT_DIR, '2_CycleGAN', 'generate_images', 'synthetic_images', 'B', f.replace('.tif', '-' + str(i) + '_synthetic.tif'))
        imgTiles = np.asarray(imgTiles, dtype='float32')
        imgTiles = imgTiles.reshape([imgTiles.shape[0], imgTiles.shape[1], imgTiles.shape[2], 1]) / 255.0
        imgStitched = stitchImage(imgTiles, IMAGE_SIZE_W, IMAGE_SIZE_H, TILE_SIZE_W, TILE_SIZE_H, manageOverlapsMode=2, return8BitImage=False)
        imgStitched = np.asarray((imgStitched >= 0.5) * 255, dtype='uint8')
        imageio.imwrite(os.path.join(ROOT_DIR, '2_CycleGAN', 'generate_images', 'synthetic_images', 'B_stitched_unfiltered', f), imgStitched)

    # Step 5: Filter Artifact Particles
    imgPath = os.path.join(ROOT_DIR, '2_CycleGAN', 'generate_images', 'synthetic_images', 'A')
    mskPath = os.path.join(ROOT_DIR, '2_CycleGAN', 'generate_images', 'synthetic_images', 'testB')
    outPath = os.path.join(ROOT_DIR, '2_CycleGAN', 'generate_images', 'synthetic_images', 'testB_Filtered')
    filterGANMasks(imgPath, mskPath, outPath, doWatershedAndFourConnectivity=False)

    imgPath = INPUT_DIR_IMAGES
    mskPath = os.path.join(ROOT_DIR, '2_CycleGAN', 'generate_images', 'synthetic_images', 'B_stitched_unfiltered')
    outPath = OUTPUT_DIR_CYCLEGAN
    filterGANMasks(imgPath, mskPath, outPath, doWatershedAndFourConnectivity=True)

    # Step 6: Train UNet
    imgPath = os.path.join(ROOT_DIR, '2_CycleGAN', 'generate_images', 'synthetic_images', 'A')
    mskPath = os.path.join(ROOT_DIR, '2_CycleGAN', 'generate_images', 'synthetic_images', 'testB_Filtered')
    
    # uNET = UNet_Segmentation.segmentationNetwork(ROOT_DIR, IMAGE_DIR, MASK_DIR, SOURCE, IMAGE_SIZE_W, IMAGE_SIZE_H)
    # uNET.runTraining()
    # Workaround for tensorflow not freeing GPU memory on RTX cards after process completion: start in own interpreter
    os.system('python UNet_Segmentation.py {} {} {} GAN --image_width={} --image_height={} --mode=training'.format(ROOT_DIR, imgPath, mskPath, TILE_SIZE_W, TILE_SIZE_H))

    # Use trained UNet for predicting segmentation masks
    # Tile SEM Images to correct Size
    for f in os.listdir(INPUT_DIR_IMAGES):
        inputImg = np.asarray(imageio.imread(os.path.join(INPUT_DIR_IMAGES, f)), dtype='float32').copy()
        imgTiles = np.asarray(tileImage(inputImg, IMAGE_SIZE_W, IMAGE_SIZE_H, TILE_SIZE_W, TILE_SIZE_H, normalizeOutput=True), dtype='float32')
        for i, imgTile in enumerate(imgTiles):
            # Save all tiles for later inference
            imageio.imwrite(os.path.join(ROOT_DIR, '3_Unet', 'Input', f.replace('.tif', '-' + str(i) + '.tif')), imgTile[:, :, 0])

    imgPath = os.path.join(ROOT_DIR, '3_Unet', 'Input')
    mskPath = os.path.join(ROOT_DIR, '3_Unet', 'Output')
    modelPath = os.path.join(ROOT_DIR, '3_Unet', 'Models', 'Model.h5')
    os.system('python UNet_Segmentation.py {} {} {} SEM --image_width={} --image_height={} --mode=inference --model_path={}'.format(ROOT_DIR, imgPath, mskPath, TILE_SIZE_W, TILE_SIZE_H, modelPath))

    # Stitch image tiles back together
    for f in os.listdir(INPUT_DIR_IMAGES):
        i = 0
        imgTiles = []
        g = os.path.join(mskPath, f.replace('.tif', '-' + str(i) + '.tif'))
        while os.path.exists(g):
            imgTiles.append(np.asarray(imageio.imread(g), dtype='float32').copy())
            i += 1
            g = os.path.join(mskPath, f.replace('.tif', '-' + str(i) + '.tif'))
        imgTiles = np.asarray(imgTiles, dtype='float32')
        imgTiles = imgTiles.reshape([imgTiles.shape[0], imgTiles.shape[1], imgTiles.shape[2], 1])
        imgStitched = stitchImage(imgTiles, IMAGE_SIZE_W, IMAGE_SIZE_H, TILE_SIZE_W, TILE_SIZE_H, manageOverlapsMode=2, return8BitImage=False)[:, :, 0]
        imageio.imwrite(os.path.join(ROOT_DIR, 'Output_Masks_UNet', f), imgStitched)
        imgStitched = segment(imgStitched, 0.5, True)
        imgStitched = eightToFourConnected(imgStitched)
        imageio.imwrite(os.path.join(ROOT_DIR, 'Output_Masks_UNet', f.replace('.tif', '_binary.tif')), imgStitched)

    print('Done: ' + str(datetime.now()))

