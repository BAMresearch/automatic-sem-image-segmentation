import os
import imageio
import numpy as np
import cv2

import multiprocessing as mp

from Measurements import Measure

from skimage.morphology import watershed
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu
from scipy import ndimage

ROOT_DIR = os.path.abspath("./")
INPUT_DIR = os.path.join(ROOT_DIR, "Masks_Predicted")
IMAGE_DIR = os.path.join(ROOT_DIR, "Images")
GROUNDTRUTH_DIR = os.path.join(ROOT_DIR, "TiO2_Masks_Manual_4connected")
WATERSHED = True
FILTER = False
RESCALING = 1
THREADS = mp.cpu_count()-2

inputDirectories = []
groundTruthImages = []

for dir in os.listdir(INPUT_DIR):
    inputDirectories.append(os.path.join(INPUT_DIR, dir))

for file in os.listdir(GROUNDTRUTH_DIR):
    if ".tif" in file:
        groundTruthImages.append(os.path.join(GROUNDTRUTH_DIR, file))


def segment(image, threshold, doWatershed=True):
    MIN_DISTANCE = 9

    img = image.copy()
    # Threshold the image if it is not already a bool
    if not img.dtype == np.bool:
        if threshold < 0:
            threshold = threshold_otsu(img)
        mask = img > threshold
    else:
        mask = img

    # If all values are the same after thresholding, skip watershed and return image directly
    if np.min(mask) == np.max(mask):
        return np.asarray(mask > 0, dtype='uint8')

    if not doWatershed:
        return np.asarray(mask > 0, dtype='uint8')

    # Get the Euclidian Distance Map and smooth it
    distance = ndimage.distance_transform_edt(mask)
    distance = ndimage.gaussian_filter(distance, sigma=1)

    # Get the local maxima of the distance map
    local_maxi = peak_local_max(distance, indices=False, min_distance=MIN_DISTANCE)

    markers = ndimage.label(local_maxi)[0]
    labels = watershed(-distance, markers, mask=mask, watershed_line=doWatershed)
    labels = labels > 0
    labels = ndimage.binary_fill_holes(labels, structure=np.ones((3, 3)))

    return np.asarray(labels * 1.0, dtype='uint8')


def calculateWholeImageIoU(image1, image2):
    return np.sum(np.logical_and(image1, image2)) / np.sum(np.logical_or(image1, image2))


def calculateInstanceIoU(image1, image2, minArea = 0):
    contours1, contourHierarchy1 = cv2.findContours(image=image1, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)
    contours2, contourHierarchy2 = cv2.findContours(image=image2, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)

    instanceIoU = []
    totalIoU = 0
    blank = np.zeros(image1.shape, dtype='uint8')
    for i in range(0, len(contours1)):
        curIoU = 0
        img1 = cv2.drawContours(blank.copy(), contours1, i, 1, cv2.FILLED)
        x1 = [contours1[i][k][0][0] for k in range(0, len(contours1[i]))]
        y1 = [contours1[i][k][0][1] for k in range(0, len(contours1[i]))]

        # Filter out false positives (i.e., everything with an area smaller than minArea)
        if polygon_area(np.asarray(x1), np.asarray(y1)) > minArea:
            for j in range(0, len(contours2)):
                x2 = [contours2[j][k][0][0] for k in range(0, len(contours2[j]))]
                y2 = [contours2[j][k][0][1] for k in range(0, len(contours2[j]))]

                # Check if axis-aligned bounding boxes overlap:
                if not (min(x2) > max(x1) or max(x2) < min(x1) or min(y2) > max(y1) or max(y2) < min(y1)):
                    img2 = cv2.drawContours(blank.copy(), contours2, j, 1, cv2.FILLED)
                    tmp = calculateWholeImageIoU(img1, img2)
                    if tmp > curIoU:
                        curIoU = tmp

            instanceIoU.append(curIoU)
            totalIoU += curIoU
    if len(instanceIoU) == 0:
        return 0
    else:
        return totalIoU/len(instanceIoU)


def ROC(predicted, groundTruth):
    TP = 0.0
    TN = 0.0
    FP = 0.0
    FN = 0.0
    for i in range(0, predicted.shape[0]):
        for j in range(0, predicted.shape[1]):
            if predicted[i, j] > groundTruth[i, j]:
                FP += 1
            elif predicted[i, j] < groundTruth[i, j]:
                FN += 1
            elif predicted[i, j] == groundTruth[i, j] and predicted[i, j] == 0:
                TN += 1
            elif predicted[i, j] == groundTruth[i, j] and predicted[i, j] == 1:
                TP += 1

    TPR = 0  # recall
    TNR = 0  # specificity
    FPR = 0  # fallout
    FNR = 0  # missrate
    if TP+FN > 0:
        TPR = TP/(TP+FN)
    if TN+FP > 0:
        TNR = TN/(TN+FP)
    if TN+FP > 0:
        FPR = FP/(TN+FP)
    if TP+FN > 0:
        FNR = FN/(TP+FN)

    return TPR, TNR, FPR, FNR


def polygon_area(x, y):
    """
    Implementation of the shoelace formula
    https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates
    :param x: List of x coordinates of the polygon
    :param y: List of x coordinates of the polygon
    :return: Area of the polygon
    """
    x_ = x - x.mean()
    y_ = y - y.mean()
    correction = x_[-1] * y_[0] - y_[-1] * x_[0]
    main_area = np.dot(x_[:-1], y_[1:]) - np.dot(y_[:-1], x_[1:])
    return 0.5 * np.abs(main_area + correction)

def eightToFourConnected(img):
    if np.count_nonzero(img) < 2 or np.count_nonzero(img) > img.size - 2: # If there are less than two 0 or 1 entries just return the image as is
        return img

    for x in range(0, img.shape[0]-1):
        for y in range(0, img.shape[1]-1):
            if img[x, y] == 0 and img[x+1, y+1] == 0 and img[x+1, y] != 0 and img[x, y+1] != 0:
                img[x+1, y] = 0
            elif img[x+1, y] == 0 and img[x, y+1] == 0 and img[x, y] != 0 and img[x+1, y+1] != 0:
                img[x, y] = 0
    return img

def applyMeanFilter(img, mask):
        m = Measure(mask, darkBackground=True, threshold=0.5, applyWatershed=False, excludeEdges=False, grayscaleImage=img)
        m.calculateMeanIntensities()
        m.filterResults('meanIntensity', np.mean(img))

        contours = np.zeros(img.shape, dtype='uint8')
        cv2.drawContours(image=contours, contours=m.contours, contourIdx=-1, color=(255, 255, 255), thickness=-1)

        return contours


def calculateROC(dir):
    print('\nCalculating ROC Curve for ' + str(dir.replace(INPUT_DIR, '')[1:]) + ':')
    avgTP = [0.0 for _ in range(0, 11)]
    avgTN = [0.0 for _ in range(0, 11)]
    avgFP = [0.0 for _ in range(0, 11)]
    avgFN = [0.0 for _ in range(0, 11)]
    avgYI = [0.0 for _ in range(0, 11)]
    for f in groundTruthImages:
        groundTruth = np.asarray(imageio.imread(f), dtype='uint8').copy()
        groundTruth = groundTruth//np.max(groundTruth)
        image = np.asarray(imageio.imread(f.replace(GROUNDTRUTH_DIR, dir).replace('_m', '')), dtype='float32').copy()
        if np.max(image) > 1.0:
            image /= 255.0
        if RESCALING != 1:
            image = cv2.resize(np.asarray(image, dtype='float32'), dsize=None, fx=RESCALING, fy=RESCALING, interpolation=cv2.INTER_CUBIC)
            if np.min(image) < 0:
                image += abs(np.min(image))
            if np.max(image) > 1:
                image /= abs(np.max(image))

        for t in range(0, 11):
            # Run watershed segmentation on input image
            THRESHOLD = t/10.0
            seg = segment(image, threshold=THRESHOLD, doWatershed=WATERSHED)
            seg = eightToFourConnected(seg)

            if FILTER:
                img = np.asarray(imageio.imread(f.replace(GROUNDTRUTH_DIR, IMAGE_DIR).replace('_m', '')), dtype='float32').copy()[0:712, 0:1024]
                seg = applyMeanFilter(img, seg)
                if np.max(seg) > 0:
                    seg = seg//np.max(seg)

            TP, TN, FP, FN = ROC(seg, groundTruth)
            print('Model ' + str(dir.replace(INPUT_DIR, '')[1:]) + ' ROC values for Image ' + str(f.replace(GROUNDTRUTH_DIR, '')[1:]) + ' at Threshold ' + str(THRESHOLD) + ': True Positives: ' + str(TP) + ', True Negatives: ' + str(TN) + ', False Positives: ' + str(FP) + ', False Negatives: ' + str(FN))
            avgTP[t] += TP / float(len(groundTruthImages))
            avgTN[t] += TN / float(len(groundTruthImages))
            avgFP[t] += FP / float(len(groundTruthImages))
            avgFN[t] += FN / float(len(groundTruthImages))
            avgYI[t] += (TP + TN - 1) / float(len(groundTruthImages))

    ROCValues = ""
    for t in range (0, 11):
        ROCValues += (str(dir.replace(INPUT_DIR, '')[1:]) + ';' + str(t/10.0) + ";" + str(avgTP[t]) + ';' + str(avgTN[t]) + ';' + str(avgFP[t]) + ';' + str(avgFN[t]) + ";" + str(avgYI[t]) + '\n')
    return (ROCValues+'\n')

def calculateIoU(dir):
    print('\nCalculating IoU Scores for ' + str(dir.replace(INPUT_DIR, '')[1:]) + ':')
    avgIoUWhole = [0.0 for i in range(0, 11)]
    avgIoUInstanceAll = [0.0 for i in range(0, 11)]
    avgIoUInstanceFiltered = [0.0 for i in range(0, 11)]
    for f in groundTruthImages:
        groundTruth = np.asarray(imageio.imread(f), dtype='uint8').copy()
        groundTruth = groundTruth//np.max(groundTruth)
        image = np.asarray(imageio.imread(f.replace(GROUNDTRUTH_DIR, dir).replace('_m', '')), dtype='float32').copy()
        if np.max(image) > 1.0:
            image /= 255.0
        if RESCALING != 1:
            groundTruth = cv2.resize(np.asarray(groundTruth, dtype='float32'), dsize=None, fx=1.0/RESCALING, fy=1.0/RESCALING, interpolation=cv2.INTER_AREA)
            groundTruth = np.asarray(groundTruth > 0.5, dtype='uint8')

        for thresh in range(0, 11):
            # Run watershed segmentation on input image
            THRESHOLD = thresh/10.0
            seg = segment(image, threshold=THRESHOLD, doWatershed=WATERSHED)
            seg = eightToFourConnected(seg)

            if FILTER:
                img = np.asarray(imageio.imread(f.replace(GROUNDTRUTH_DIR, IMAGE_DIR).replace('_m', '')), dtype='float32').copy()
                seg = applyMeanFilter(img, seg)
                if np.max(seg) > 0:
                    seg = seg//np.max(seg)

            IoUWhole = calculateWholeImageIoU(seg, groundTruth)
            IoUInstanceAll = calculateInstanceIoU(seg, groundTruth, 0)
            IoUInstanceFiltered = calculateInstanceIoU(seg, groundTruth, 9)
            print('Model ' + str(dir.replace(INPUT_DIR, '')[1:]) + ' IoU score for Image ' + str(f.replace(GROUNDTRUTH_DIR + '\\', '')) + ' at Threshold ' + str(THRESHOLD) + ': Whole Image: ' + str(IoUWhole) + '; Instances (all): ' + str(IoUInstanceAll) + '; Instances (Area > 9 sq.pixel): ' + str(IoUInstanceFiltered))
            avgIoUWhole[thresh-1] += IoUWhole / float(len(groundTruthImages))
            avgIoUInstanceAll[thresh-1] += IoUInstanceAll / float(len(groundTruthImages))
            avgIoUInstanceFiltered[thresh-1] += IoUInstanceFiltered / float(len(groundTruthImages))
    avgIoUWholeBest = 0.0
    avgIoUInstanceAllBest = 0.0
    avgIoUInstanceFilteredBest = 0.0
    bestThresholdWhole = 0.0
    bestThresholdInstanceAll = 0.0
    bestThresholdInstanceFiltered = 0.0
    for i in range(0, len(avgIoUWhole)):
        if avgIoUWhole[i] > avgIoUWholeBest:
            avgIoUWholeBest = avgIoUWhole[i]
            bestThresholdWhole = i/10.0
        if avgIoUInstanceAll[i] > avgIoUInstanceAllBest:
            avgIoUInstanceAllBest = avgIoUInstanceAll[i]
            bestThresholdInstanceAll = i/10.0
        if avgIoUInstanceFiltered[i] > avgIoUInstanceFilteredBest:
            avgIoUInstanceFilteredBest = avgIoUInstanceFiltered[i]
            bestThresholdInstanceFiltered = i/10.0
    print('Model ' + str(dir.replace(INPUT_DIR, '')[1:]) + ' average IoU scores: Whole Image: ' + str(avgIoUWholeBest) + ' at Threshold ' + str(bestThresholdWhole) + '; Instances (all): ' + str(avgIoUInstanceAllBest) + ' at Threshold ' + str(bestThresholdInstanceAll) + '; Instances (Area > 9 sq.pixel): ' + str(avgIoUInstanceFilteredBest) + ' at Threshold ' + str(bestThresholdInstanceFiltered))
    return (str(dir.replace(INPUT_DIR, '')[1:]) + ';' + str(avgIoUWholeBest) + ';' + str(bestThresholdWhole) + ';' + str(avgIoUInstanceAllBest) + ';' + str(bestThresholdInstanceAll) + ';' + str(avgIoUInstanceFilteredBest) + ';' + str(bestThresholdInstanceFiltered) + '\n')


if __name__ == '__main__':
    pool = mp.Pool(THREADS)

    result_objectsIoU = [pool.map_async(calculateIoU, [f for f in inputDirectories])]
    result_objectsROC = [pool.map_async(calculateROC, [f for f in inputDirectories])]

    pool.close()
    pool.join()

    IoUScores = [r.get() for r in result_objectsIoU][0]
    ROCValues = [r.get() for r in result_objectsROC][0]

    print('\n\n\n------------')
    fName = 'IoUScores_Variable_Threshold'
    if WATERSHED:
        fName += '_Watershed'
    else:
        fName += '_No_Watershed'

    if FILTER:
        fName += '_Filtered.csv'
    else:
        fName += '.csv'

    f = open(os.path.join(ROOT_DIR, fName), 'a')
    f.write('Model;Average IoU score (Whole Image);At Threshold;Average IoU score (All Instances);At Threshold;Average IoU score (Instances > 9 sq.pixel);At Threshold\n')
    print('Model;Average IoU score (Whole Image);At Threshold;Average IoU score (All Instances);At Threshold;Average IoU score (Instances > 9 sq.pixel);At Threshold\n')
    for i in IoUScores:
        f.write(str(i))
        print(str(i))
    f.close()

    print('\n\n\n------------')
    fName = 'ROCValues_Variable_Threshold'
    if WATERSHED:
        fName += '_Watershed'
    else:
        fName += '_No_Watershed'

    if FILTER:
        fName += '_Filtered.csv'
    else:
        fName += '.csv'

    f = open(os.path.join(ROOT_DIR, fName), 'a')
    f.write('Model;Threshold;True Positves;True Negatives;False Positives;False Negatives;Youdens Index\n')
    print('Model;Threshold;True Positves;True Negatives;False Positives;False Negatives;Youdens Index\n')
    for i in ROCValues:
        f.write(str(i))
        print(str(i))
    f.close()

    bestYoudenIndices = []
    maxValue = 0.0
    atThreshold = 0.0
    modelName = ''
    for i in ROCValues:
        j = i.split('\n')
        for k in j:
            tmp = k.split(';')
            if len(tmp) > 5:
                if float(tmp[6]) > maxValue:
                    maxValue = float(tmp[6])
                    atThreshold = float(tmp[1])
                    modelName = str(tmp[0])
        bestYoudenIndices.append(str(modelName) + ';' + str(maxValue) +';' + str(atThreshold) + '\n')
        maxValue = 0.0
        atThreshold = 0.0
        modelName = ''


    f = open(os.path.join(ROOT_DIR, fName), 'a')
    f.write('\n\nModel;Best Youdens Index;At Threshold\n')
    print('\n\nModel;Best Youdens Index;At Threshold\n')
    for i in bestYoudenIndices:
        f.write(str(i))
        print(str(i))
    f.close()
