import os
import imageio
import numpy as np
import math

from datetime import datetime
import time

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler
import tensorflow.keras.backend as K

from MultiResUNet import MultiResUNet

import argparse


class myDataset():
    def __init__(self, IMAGE_DIR, MASK_DIR, IMAGE_SIZE_W, IMAGE_SIZE_H, *args):
        self.image_ids = []
        self.image_info = {}
        self.type = ''
        self.source = ''
        self.IMAGE_DIR = IMAGE_DIR
        self.MASK_DIR = MASK_DIR
        self.IMAGE_SIZE_W = IMAGE_SIZE_W
        self.IMAGE_SIZE_H = IMAGE_SIZE_H

    def add_image(self, image_id, path, **kwargs):
        self.image_info[image_id] = {'id': image_id, 'path': path, **kwargs}
        self.image_ids.append(image_id)

    def initialize_images(self, subset, source):
        """Load a subset of the image dataset.
        subset: Subset to load: train, val, or all
        source: Are the masks simulated by a GAN?
        """

        # Add images
        allImages = [os.path.join(self.IMAGE_DIR, f) for f in os.listdir(self.IMAGE_DIR)]

        # Train or validation dataset?
        assert subset in ["train", "val", "all"]

        self.source = source
        self.type = subset

        if (subset == "train"):
            allImages = allImages[:int(0.8*len(allImages))]

        elif (subset == "val"):
            allImages = allImages[int(0.8 * len(allImages)):]

        for i in range(0, len(allImages)):
            image_path = allImages[i]
            if self.source == 'GAN':
                mask_path = image_path.replace(self.IMAGE_DIR, self.MASK_DIR).replace("_synthetic.tif", ".tif")
            else:
                mask_path = image_path.replace(self.IMAGE_DIR, self.MASK_DIR)

            if self.source == 'SEM' or self.source == 'TSEM':
                for j in range(0, 4):
                    for k in range(0, 4):
                        self.add_image(
                            image_id='{:04d}'.format(i) + '_tile_' + str(j) + '_augmentation_' + str(k),
                            path=image_path,
                            width=self.IMAGE_SIZE_W,
                            height=self.IMAGE_SIZE_H,
                            mask=mask_path,
                            tile=j,
                            augmentation=k,
                            source=source
                        )
            else:
                for k in range(0, 4):
                    self.add_image(
                        image_id='{:04d}'.format(i) + '_tile_' + str(-1) + '_augmentation_' + str(k),
                        path=image_path,
                        width=self.IMAGE_SIZE_W,
                        height=self.IMAGE_SIZE_H,
                        mask=mask_path,
                        tile=-1,
                        augmentation=k,
                        source=source
                    )

    def load_image(self, image_id):
        """
        Load the image from file
        """
        info = self.image_info[image_id]
        tile = info['tile']
        augmentation = info['augmentation']
        source = info['source']

        # Load image
        image = np.asarray(imageio.imread(info['path']), dtype='float32').copy()

        image -= np.min(image)
        image /= np.max(image)

        if source == 'SEM' or source == 'TSEM':
            # Crop to tile size
            if tile == 0:
                image = image[0:self.IMAGE_SIZE_H // 2, 0:self.IMAGE_SIZE_W // 2]
            elif tile == 1:
                image = image[self.IMAGE_SIZE_H // 2:self.IMAGE_SIZE_H, 0:self.IMAGE_SIZE_W // 2]
            elif tile == 2:
                image = image[0:self.IMAGE_SIZE_H // 2, self.IMAGE_SIZE_W // 2:self.IMAGE_SIZE_W]
            elif tile == 3:
                image = image[self.IMAGE_SIZE_H // 2:self.IMAGE_SIZE_H, self.IMAGE_SIZE_W // 2:self.IMAGE_SIZE_W]

        if augmentation == 1:
            image = np.fliplr(image)
        elif augmentation == 2:
            image = np.flipud(image)
        elif augmentation == 3:
            image = np.fliplr(np.flipud(image))

        return image

    def load_mask(self, image_id):
        """
        Load the mask from file
        """

        info = self.image_info[image_id]
        tile = info['tile']
        augmentation = info['augmentation']
        source = info['source']

        # Load the Masks from files
        mask = np.asarray(imageio.imread(info['mask']), dtype='uint8').copy() // 255

        if source == 'SEM' or source == 'TSEM':
            # Crop to tile size
            if tile == 0:
                mask = mask[0:self.IMAGE_SIZE_H // 2, 0:self.IMAGE_SIZE_W // 2]
            elif tile == 1:
                mask = mask[self.IMAGE_SIZE_H // 2:self.IMAGE_SIZE_H, 0:self.IMAGE_SIZE_W // 2]
            elif tile == 2:
                mask = mask[0:self.IMAGE_SIZE_H // 2, self.IMAGE_SIZE_W // 2:self.IMAGE_SIZE_W]
            elif tile == 3:
                mask = mask[self.IMAGE_SIZE_H // 2:self.IMAGE_SIZE_H, self.IMAGE_SIZE_W // 2:self.IMAGE_SIZE_W]

        if augmentation == 1:
            mask = np.fliplr(mask)
        elif augmentation == 2:
            mask = np.flipud(mask)
        elif augmentation == 3:
            mask = np.fliplr(np.flipud(mask))

        return mask


class segmentationNetwork():
    def __init__(self, ROOT_DIR, IMAGE_DIR, MASK_DIR, SOURCE, IMAGE_SIZE_W=512, IMAGE_SIZE_H=352):
        # Root directory of the project
        self.ROOT_DIR = os.path.join(ROOT_DIR, '3_UNet')

        # Directory to save logs and trained model
        self.MODEL_DIR = os.path.join(self.ROOT_DIR, "Models")

        # Path to training images and masks
        self.IMAGE_DIR = IMAGE_DIR
        self.MASK_DIR = MASK_DIR

        # Set up global variables
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

        # Hyper Parameters and Configuration
        self.BATCH_SIZE = 2
        self.EPOCHS = 100
        self.LEARNING_RATE = 0.001
        self.LOSS_FUNCTION = 'binary_crossentropy'
        self.LR_DECAY = 'STEP_DECAY'
        self.IMAGE_SIZE_W = IMAGE_SIZE_W
        self.IMAGE_SIZE_H = IMAGE_SIZE_H
        self.FILTERS = 16

        # Initialize Dataset
        assert SOURCE in ['SEM', 'TSEM', 'GAN']
        self.SOURCE = SOURCE
        self.dataset_train = myDataset(self.IMAGE_DIR, self.MASK_DIR, self.IMAGE_SIZE_W, self.IMAGE_SIZE_H)
        self.dataset_val = myDataset(self.IMAGE_DIR, self.MASK_DIR, self.IMAGE_SIZE_W, self.IMAGE_SIZE_H)
        self.dataset_train.initialize_images('train', self.SOURCE)
        self.dataset_val.initialize_images('val', self.SOURCE)

    def loadImages(self, subset):
        assert subset in ['train', 'val']

        if subset == "train":
            limit = len(self.dataset_train.image_ids)
            print('Importing ' + str(limit) + ' training images: ' + str(datetime.now()))
            X = np.array([self.dataset_train.load_image(self.dataset_train.image_ids[index]) for index in range(0, limit)])
            Y = np.array([self.dataset_train.load_mask(self.dataset_train.image_ids[index]) for index in range(0, limit)])
            print(str(limit) + ' training images successfully imported: ' + str(datetime.now()))
        elif subset == "val":
            limit = len(self.dataset_val.image_ids)
            print('Importing ' + str(limit) + ' validation images: ' + str(datetime.now()))
            X = np.array([self.dataset_val.load_image(self.dataset_val.image_ids[index]) for index in range(0, limit)])
            Y = np.array([self.dataset_val.load_mask(self.dataset_val.image_ids[index]) for index in range(0, limit)])
            print(str(limit) + ' validation images successfully imported: ' + str(datetime.now()))
        return (X.reshape([X.shape[0], X.shape[1], X.shape[2], 1]), Y.reshape([Y.shape[0], Y.shape[1], Y.shape[2], 1]))


    def stepDecay(self, epoch):
       initial_lrate = self.LEARNING_RATE
       drop = 0.5
       epochs_drop = 10.0
       lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
       return lrate

    def linearDecay(self, epoch):
        POWER = 1  # 1 -> Linear Decay
        initial_lrate = self.LEARNING_RATE
        decay = (1 - (epoch / float(self.EPOCHS))) ** POWER
        lrate = initial_lrate * decay
        # return the new learning rate
        return lrate

    def runTraining(self):
        # Load the Training and Validation Datasets
        X_train, Y_train = self.loadImages('train')
        X_val, Y_val = self.loadImages('val')

        name = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime()) + '_Masks_' + self.SOURCE
        CHECKPOINT_PATH = os.path.join(self.MODEL_DIR, name + "_checkpoint.h5")
        LOG_PATH = os.path.join(self.MODEL_DIR, name + "_log.csv")
        checkpoint = ModelCheckpoint(CHECKPOINT_PATH, monitor='loss', verbose=1, save_best_only=True, mode='min')
        csvLogger = CSVLogger(LOG_PATH, separator=';', append=True)
        callbacks_list = [checkpoint, csvLogger]
        if self.LR_DECAY == 'STEP_DECAY':
            rateScheduler = LearningRateScheduler(self.stepDecay)
            callbacks_list.append(rateScheduler)
        elif self.LR_DECAY == 'LINEAR_DECAY':
            rateScheduler = LearningRateScheduler(self.linearDecay)
            callbacks_list.append(rateScheduler)


        print('Initializing neural network')

        if self.SOURCE == 'SEM' or self.SOURCE == 'TSEM':
            input_img = Input(shape=(self.IMAGE_SIZE_H // 2, self.IMAGE_SIZE_W // 2, 1))
        else:
            input_img = Input(shape=(self.IMAGE_SIZE_H, self.IMAGE_SIZE_W, 1))

        m = MultiResUNet.MultiResUnet(input_img, outputChannels=1, convFilters=self.FILTERS)
        myModel = Model(input_img, m)  # Method imported from keras.models

        os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # model will be trained on GPU 0

        # Custom weight function for class weight balancing
        weighting = (Y_train.size - np.count_nonzero(Y_train)) / np.count_nonzero(Y_train)

        def weighted_bce(y_true, y_pred):
            weights = (y_true * (weighting-1)) + 1
            bce = K.binary_crossentropy(y_true, y_pred)
            weighted_loss = K.mean(bce * weights)
            return weighted_loss

        if isinstance(self.LR_DECAY, float):
            opt = Adam(lr=self.LEARNING_RATE, decay=self.LR_DECAY)
        else:
            opt = Adam(lr=self.LEARNING_RATE, decay=0.0)

        myModel.compile(loss=weighted_bce, optimizer=opt, metrics=['mae', 'acc'])

        # Train the Model
        print('Start training the model: ' + str(datetime.now()))
        INITIAL_EPOCH = int(K.get_value(myModel.optimizer.iterations)/X_train.shape[0]*self.BATCH_SIZE)

        myModel_train = myModel.fit(X_train,
                                    Y_train,
                                    batch_size=self.BATCH_SIZE,
                                    epochs=self.EPOCHS,
                                    initial_epoch=INITIAL_EPOCH,
                                    verbose=1,
                                    callbacks=callbacks_list,
                                    validation_data=(X_val, Y_val))

        # Save the model
        print('Saving model: ' + str(os.path.join(self.MODEL_DIR, 'Model.h5')))
        myModel.save(os.path.join(self.MODEL_DIR, 'Model.h5'))
        return myModel_train

    @classmethod
    def loadModel(self, modelPath):
        weighting = 1

        def weighted_bce(y_true, y_pred):
            weights = (y_true * (weighting - 1)) + 1
            bce = K.binary_crossentropy(y_true, y_pred)
            weighted_bce = K.mean(bce * weights)
            return weighted_bce

        return load_model(modelPath, custom_objects={'weighted_bce': weighted_bce})

    @classmethod
    def runInference(self, model, IMAGE_DIR, MASK_DIR):
        for f in os.listdir(IMAGE_DIR):
            imgTile = np.asarray(imageio.imread(os.path.join(IMAGE_DIR, f)), dtype='float32')
            imgTile -= np.min(imgTile)
            imgTile /= np.max(imgTile)
            imgTile = imgTile.reshape((1, imgTile.shape[0], imgTile.shape[1], 1))
            pred = model.predict(imgTile)
            imageio.imwrite(os.path.join(MASK_DIR, f), pred[0, :, :, 0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MultiRes UNet for image segmentation.')
    parser.add_argument('root_dir', type=str, help='Root directory where the files should be stored.')
    parser.add_argument('image_dir', type=str, help='Directory where the images are stored.')
    parser.add_argument('mask_dir', type=str, help='Directory where the masks are stored.')
    parser.add_argument('source', type=str, choices=['SEM', 'TSEM', 'GAN'], help='Source of the masks [SEM, TSEM, GAN].')
    parser.add_argument('--image_width', type=int, default=512, help='Image width')
    parser.add_argument('--image_height', type=int, default=352, help='Image height')
    parser.add_argument('--mode', type=str, choices=['training', 'inference'], default='training', help='Whether to use the network in training or in inference mode.')
    parser.add_argument('--model_path', type=str, default='Model.h5', help='Name of the trained model (only necessary for inference mode).')

    args = parser.parse_args()
    ROOT_DIR = args.root_dir
    IMAGE_DIR = args.image_dir
    MASK_DIR = args.mask_dir
    SOURCE = args.source
    IMAGE_SIZE_W = args.image_width
    IMAGE_SIZE_H = args.image_height
    MODE = args.mode
    MODEL_PATH = args.model_path

    if MODE == 'training':
        uNET = segmentationNetwork(ROOT_DIR, IMAGE_DIR, MASK_DIR, SOURCE, IMAGE_SIZE_W, IMAGE_SIZE_H)
        uNET.runTraining()
    elif MODE == 'inference':
        model = segmentationNetwork.loadModel(MODEL_PATH)
        segmentationNetwork.runInference(model, IMAGE_DIR, MASK_DIR)
