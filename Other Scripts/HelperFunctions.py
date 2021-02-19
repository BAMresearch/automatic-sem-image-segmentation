import numpy as np
# import tensorflow.keras.backend as K
import math

from skimage.morphology import watershed
from skimage.feature import peak_local_max
from scipy import ndimage
from skimage.filters import threshold_otsu


# Function for watershed segmentation
def segment(image, threshold, watershedLines):
    MIN_DISTANCE = 9

    img = image[:, :, 0]
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

    markers = ndimage.label(local_maxi)[0]
    labels = watershed(-distance, markers, mask=mask, watershed_line=watershedLines)
    labels = labels > 0
    labels = ndimage.binary_fill_holes(labels, structure=np.ones((3, 3)))

    return EightToFourConnected(np.asarray((labels > 0) * 255, dtype='uint8'))


# Function for preprocessing and tiling an image (in case it is too large for the neural network model)
def preprocessImage(img, IMAGE_SIZE_W, IMAGE_SIZE_H, INPUT_SIZE_W, INPUT_SIZE_H, minOverlap=2):
    img = img.astype(np.float32)
    # Normalize entire image
    # img -= np.min(img)
    # img /= np.max(img)
    noOfXTiles = math.ceil(IMAGE_SIZE_W / INPUT_SIZE_W)
    noOfYTiles = math.ceil(IMAGE_SIZE_H / INPUT_SIZE_H)

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
            offsetX = math.ceil(i * (INPUT_SIZE_W - ((INPUT_SIZE_W * noOfXTiles - IMAGE_SIZE_W) / (noOfXTiles - 1))))
        else:
            offsetX = 0

        for j in range(0, noOfYTiles):
            if (noOfYTiles > 1):
                offsetY = math.ceil(
                    j * (INPUT_SIZE_H - ((INPUT_SIZE_H * noOfYTiles - IMAGE_SIZE_H) / (noOfYTiles - 1))))
            else:
                offsetY = 0

            imgTiles[k, :, :, 0] = img[offsetY:min(offsetY + INPUT_SIZE_H, IMAGE_SIZE_H), offsetX:min(offsetX + INPUT_SIZE_W, IMAGE_SIZE_W)]

            # Normalize each tile individually
            imgTiles[k, :, :, 0] -= np.min(imgTiles[k, :, :, 0])
            imgTiles[k, :, :, 0] /= np.max(imgTiles[k, :, :, 0])

            k += 1

    return imgTiles


# Function for deprocessing and reassembling the neural network output into an image (in case it
# contains a batch of tiles, they are reassembled into a seamless image by taking the maximum (manageOverlapsMode=1),
# average (manageOverlapsMode=1), or cropping (manageOverlapsMode=2) in overlapping areas)
def deprocessImage(img, IMAGE_SIZE_W, IMAGE_SIZE_H, INPUT_SIZE_W, INPUT_SIZE_H, minOverlap=2, manageOverlapsMode=2, return8BitImage=False):
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
    overlapSizeX = 0
    overlapSizeY = 0
    if noOfXTiles > 1:
        overlapSizeX = (INPUT_SIZE_W * noOfXTiles - IMAGE_SIZE_W) // (2 * (noOfXTiles - 1))
    if noOfYTiles > 1:
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
                imgStitched[offsetY:min(offsetY + INPUT_SIZE_H, IMAGE_SIZE_H),
                offsetX:min(offsetX + INPUT_SIZE_W, IMAGE_SIZE_W), :] = np.maximum(img[k, :, :, :], imgStitched[offsetY:min(offsetY + INPUT_SIZE_H, IMAGE_SIZE_H), offsetX:min(offsetX + INPUT_SIZE_W, IMAGE_SIZE_W), :])
            elif manageOverlapsMode == 1:
                # Average in overlapping regions
                imgStitched[offsetY:min(offsetY + INPUT_SIZE_H, IMAGE_SIZE_H),
                offsetX:min(offsetX + INPUT_SIZE_W, IMAGE_SIZE_W), :] += img[k, :, :, :]
                overlaps[offsetY:min(offsetY + INPUT_SIZE_H, IMAGE_SIZE_H),
                offsetX:min(offsetX + INPUT_SIZE_W, IMAGE_SIZE_W), :] += np.ones((INPUT_SIZE_H, INPUT_SIZE_W, img.shape[-1]), dtype='uint8')
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


# Change connectivity after watershed from 8 to 4
def EightToFourConnected(img):
    if np.count_nonzero(img) > 2 or np.count_nonzero(
            img) < img.size - 2:  # If there are less than two 0 or 1 entries in the entire image just use the image as is
        for x in range(0, img.shape[0] - 1):
            for y in range(0, img.shape[1] - 1):
                if img[x, y] == 0 and img[x + 1, y + 1] == 0 and img[x + 1, y] != 0 and img[x, y + 1] != 0:
                    img[x + 1, y] = 0
                elif img[x + 1, y] == 0 and img[x, y + 1] == 0 and img[x, y] != 0 and img[x + 1, y + 1] != 0:
                    img[x, y] = 0
    return img


def parseCalibration(f):
    h = open(f, 'r', encoding='ansi')
    fc = h.read().split('\n')
    h.close()
    for i in range(len(fc) - 1, -1, -1):  # start searching from the back
        if "<PixelSize>" in fc[i]:
            tmp = fc[i + 1]
            return float(tmp[tmp.find(">") + 1:tmp.rfind("<")]) * 1e9


# Custom weight functions were used for some models during training. They are not really needed for inference,
# but they still need to be declared when using the keras load_model function.
weighting = 1
def weighted_bce(y_true, y_pred):
    weights = (y_true * (weighting - 1)) + 1
    bce = K.binary_crossentropy(y_true, y_pred)
    weighted_bce = K.mean(bce * weights)
    return weighted_bce


weights = [1 for i in range(0, 3)]
def weighted_cce(y_true, y_pred):
    weighted_cce = 0.0
    for i in range(0, 3):
        weighted = (y_true[:, :, :, i] * (weights[i] - 1)) + 1
        bce = K.binary_crossentropy(y_true[:, :, :, i], y_pred[:, :, :, i])
        weighted_cce += K.mean(bce * weighted)
    return weighted_cce
