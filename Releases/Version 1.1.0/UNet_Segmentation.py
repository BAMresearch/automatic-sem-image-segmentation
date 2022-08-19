import os
import random
import numpy as np
import math
from datetime import datetime
import time
import tensorflow as tf
from tqdm import tqdm
from PIL import Image

import HelperFunctions


class ImageDataset:
    def __init__(self, image_dir, mask_dir, contrast_optimization_range=(1, 99)):
        self.image_ids = []
        self.image_info = {}
        self.type = ''
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.contrast_optimization_range = contrast_optimization_range

    def add_image(self, image_id, path, mask, augmentation):
        self.image_info[image_id] = {'id': image_id, 'image_path': path, 'mask_path': mask, 'augmentation': augmentation}
        self.image_ids.append(image_id)

    def initialize_images(self, subset, train_val_split=0.8, seed=1234):
        """Load a subset of the image dataset."""

        # Add images
        images = []
        all_images = HelperFunctions.get_image_file_paths_from_directory(self.image_dir)
        random.Random(seed).shuffle(all_images)

        # Train or validation dataset?
        assert subset in ["train", "val"]

        self.type = subset

        if subset == "train":
            images = all_images[:int(train_val_split * len(all_images))]

        elif subset == "val":
            images = all_images[int(train_val_split * len(all_images)):]

        for i in range(0, len(images)):
            image_path = images[i]
            mask_path = image_path.replace(self.image_dir, self.mask_dir)

            for j in range(0, 4):
                self.add_image(
                    image_id='{:05d}'.format(i) + '_augmentation_' + str(j),
                    path=image_path,
                    mask=mask_path,
                    augmentation=j,
                )

    def load_from_file(self, image_ids, is_mask):
        """
        Load images from file
        """
        images = []
        if isinstance(image_ids, str):
            image_ids = [image_ids, ]

        for image_id in image_ids:
            info = self.image_info[image_id]
            augmentation = info['augmentation']

            # Load image
            if is_mask:
                image = HelperFunctions.load_and_preprocess_images(info['mask_path'], normalization_range=(0, 1), threshold_value=0.5)[0]
            else:
                image = HelperFunctions.load_and_preprocess_images(info['image_path'], normalization_range=(0, 1), contrast_optimization_range=self.contrast_optimization_range)[0]

            if augmentation == 1:
                image = np.fliplr(image)
            elif augmentation == 2:
                image = np.flipud(image)
            elif augmentation == 3:
                image = np.fliplr(np.flipud(image))
            images.append(image)

        return np.asarray(images, dtype='float32')


class DataLoader(tf.keras.utils.Sequence):
    def __init__(self, dataset, batch_size=1, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.all_image_ids = self.dataset.image_ids.copy()

    def __len__(self):
        return math.ceil(len(self.all_image_ids) / self.batch_size)

    def __getitem__(self, idx):
        img_ids = self.all_image_ids[idx * self.batch_size: (idx + 1) * self.batch_size]
        return self.dataset.load_from_file(image_ids=img_ids, is_mask=False), self.dataset.load_from_file(image_ids=img_ids, is_mask=True)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.all_image_ids)


class DataSet(tf.keras.utils.Sequence):
    def __init__(self, x, y, batch_size=1, shuffle=True):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return math.ceil(self.x.shape[0] / self.batch_size)

    def __getitem__(self, idx):
        return self.x[idx * self.batch_size: (idx + 1) * self.batch_size], self.y[idx * self.batch_size: (idx + 1) * self.batch_size]

    def on_epoch_end(self):
        if self.shuffle:
            xy = list(zip(self.x, self.y))
            random.shuffle(xy)
            self.x, self.y = zip(*xy)
            self.x = np.asarray(self.x, dtype='float32')
            self.y = np.asarray(self.y, dtype='float32')


class UNet:
    def __init__(self, root_dir, image_dir, mask_dir, allow_memory_growth=True, use_gpus_no=(0, )):
        # Root directory of the project
        self.root_dir = os.path.join(root_dir, '3_UNet')

        # Directory to save logs and trained model
        self.model_dir = os.path.join(self.root_dir, "Models")

        # Path to training images and masks
        self.image_dir = image_dir
        self.mask_dir = mask_dir

        # Set up global variables
        self.use_dataloader = False
        self.contrast_optimization_range = (1, 99)
        self.prefix = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())

        # Hyper Parameters and Model Configuration
        self.batch_size = 1
        self.epochs = 100
        self.learning_rate = 0.001
        self.loss_function = 'binary_crossentropy'
        self.lr_decay = 'STEP_DECAY'  # LINEAR_DECAY or STEP_DECAY
        self.image_shape = (384, 384, 1)
        self.filters = 16
        self.output_channels = 1

        # Tensorflow configuration
        self.allow_memory_growth = allow_memory_growth
        self.use_gpus_no = use_gpus_no
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                gpus_used = [gpus[i] for i in self.use_gpus_no]
                tf.config.set_visible_devices(gpus_used, 'GPU')
                if self.allow_memory_growth:
                    for gpu in gpus_used:
                        tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
            except RuntimeError as e:
                # Visible devices must be set before GPUs have been initialized
                print(e)

        # Initialize Variables
        self.dataset_train = None
        self.dataset_val = None
        self.training_data = None
        self.validation_data = None
        self.model = None

    def load_images(self, subset):
        assert subset in ['train', 'val']
        dataset = None

        if self.use_dataloader:
            if subset == "train":
                print('--- Using data generator during training ---')
                dataset = DataLoader(self.dataset_train, self.batch_size)
            elif subset == "val":
                dataset = DataLoader(self.dataset_val, self.batch_size)
        else:
            if subset == "train":
                print(f'Importing {len(self.dataset_train.image_ids)} augmented training images: {datetime.now()}')
                x = self.dataset_train.load_from_file(self.dataset_train.image_ids, is_mask=False)
                y = self.dataset_train.load_from_file(self.dataset_train.image_ids, is_mask=True)
                dataset = DataSet(x, y, self.batch_size)
                print(f'{x.shape[0]} augmented training images successfully imported: {datetime.now()}')
            elif subset == "val":
                print(f'Importing {len(self.dataset_val.image_ids)} augmented validation images: {datetime.now()}')
                x = self.dataset_val.load_from_file(self.dataset_val.image_ids, is_mask=False)
                y = self.dataset_val.load_from_file(self.dataset_val.image_ids, is_mask=True)
                dataset = DataSet(x, y, self.batch_size)
                print(f'{x.shape[0]} augmented training images successfully imported: {datetime.now()}')

        return dataset

    def step_decay(self, epoch, drop=0.5, epochs_drop=10):
        initial_lrate = self.learning_rate
        lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
        return lrate

    def linear_decay(self, epoch):
        power = 1  # 1 -> Linear Decay
        initial_lrate = self.learning_rate
        decay = (1 - (epoch / float(self.epochs))) ** power
        lrate = initial_lrate * decay
        return lrate

    def run_training(self):
        # Load the Training and Validation Datasets
        self.dataset_train = ImageDataset(self.image_dir, self.mask_dir, self.contrast_optimization_range)
        self.dataset_val = ImageDataset(self.image_dir, self.mask_dir, self.contrast_optimization_range)
        self.dataset_train.initialize_images('train')
        self.dataset_val.initialize_images('val')

        self.training_data = self.load_images('train')
        self.validation_data = self.load_images('val')

        self.model = self.create_model()

        # Callbacks
        checkpoint_path = os.path.join(self.model_dir, self.prefix, "Checkpoint_Lowest_Loss")
        log_path = os.path.join(self.model_dir, self.prefix, 'training_log.csv')
        checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='loss', verbose=1, save_best_only=True, mode='min')
        csv_logger = tf.keras.callbacks.CSVLogger(log_path, separator=';', append=True)
        callbacks_list = [checkpoint, csv_logger]
        if self.lr_decay == 'STEP_DECAY':
            rate_scheduler = tf.keras.callbacks.LearningRateScheduler(self.step_decay)
            callbacks_list.append(rate_scheduler)
        elif self.lr_decay == 'LINEAR_DECAY':
            rate_scheduler = tf.keras.callbacks.LearningRateScheduler(self.linear_decay)
            callbacks_list.append(rate_scheduler)

        # Train the Model
        print('Start training the model: ' + str(datetime.now()))

        self.model.fit(self.training_data,
                       batch_size=self.batch_size,
                       epochs=self.epochs,
                       verbose=1,
                       callbacks=callbacks_list,
                       validation_data=self.validation_data,
                       )

        # Save the model
        print('Saving model to: ' + str(os.path.join(self.model_dir, self.prefix)))
        self.model.save(os.path.join(self.model_dir, self.prefix))
        return self.model

    def run_inference(self, files, output_directory, model=None, tile_images=False, threshold=-1, watershed_lines=True, min_distance=9, min_overlap=2, manage_overlap_mode=2, use_gpu=False):
        if self.model is None:
            # Load the most recent model
            unweighted_bce = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)
            weighting = 1  # Not used for inference, just set to 1

            def weighted_bce(y_true, y_pred):
                weights = (y_true * (weighting - 1)) + 1
                bce = tf.expand_dims(unweighted_bce(y_true, y_pred), -1)
                return tf.reduce_mean(bce * weights)

            if model is None:
                # Load the most recent model
                self.model = tf.keras.models.load_model(os.path.join(self.model_dir, os.listdir(self.model_dir)[-1]), custom_objects={'weighted_bce': weighted_bce})
            elif isinstance(self.model, str):
                # Load the specified model
                self.model = tf.keras.models.load_model(model, custom_objects={'weighted_bce': weighted_bce})
            else:
                self.model = model

        if use_gpu:
            device = tf.config.list_logical_devices('GPU')[0]
        else:
            device = tf.config.list_logical_devices('CPU')[0]

        input_files = HelperFunctions.load_and_preprocess_images(files, normalization_range=(0, 1), contrast_optimization_range=self.contrast_optimization_range)
        file_names = HelperFunctions.get_image_file_paths_from_directory(files)

        if not tile_images and input_files[0].shape != self.image_shape:
            input_img = tf.keras.layers.Input(shape=(input_files.shape[1], input_files.shape[2], 1))
            multires_unet = UNet.multi_res_unet(input_img, output_channels=1, conv_filters=self.filters)
            model_new = tf.keras.models.Model(input_img, multires_unet)
            model_new.set_weights(self.model.get_weights())
            model = model_new

        with tf.device(device):
            for i in tqdm(range(0, input_files.shape[0])):
                input_file = input_files[i]
                if tile_images:
                    tiles = np.asarray(HelperFunctions.tile_image(input_file, self.image_shape[0], self.image_shape[1], min_overlap=min_overlap))
                    prediction = np.asarray([model(img.reshape(1, img.shape[0], img.shape[1], img.shape[2]))[0].numpy() for img in tiles])
                    img = HelperFunctions.stitch_image(prediction, input_file.shape[1], input_file.shape[0], min_overlap=min_overlap, manage_overlap_mode=manage_overlap_mode)
                else:
                    prediction = model(input_file.reshape(1, input_file.shape[0], input_file.shape[1], input_file.shape[2]))
                    img = prediction[0].numpy().copy()
                img = img[:, :, 0]
                Image.fromarray(img).save(os.path.join(output_directory, os.path.split(file_names[i])[-1].replace(os.path.splitext(file_names[i])[-1], '_raw.tif')))
                img -= np.min(img)
                img /= np.max(img)
                img *= 255
                img = img.astype(np.uint8)
                img = HelperFunctions.segment(image=img, threshold=threshold, watershed_lines=watershed_lines, min_distance=min_distance, use_four_connectivity=True)
                Image.fromarray(img).save(os.path.join(output_directory, os.path.split(file_names[i])[-1]))

    def create_model(self):
        if self.use_dataloader:
            zeros = 0
            ones = 0
            tmp = None
            for index in range(0, len(self.dataset_train.image_ids)):
                tmp = np.array(self.dataset_train.load_from_file(self.dataset_train.image_ids[index], is_mask=True))
                zeros += np.count_nonzero(tmp == 0)
                ones += np.count_nonzero(tmp)
            weighting = zeros / ones
            self.image_shape = tmp.shape[1:3]
        else:
            weighting = np.count_nonzero(self.training_data.y == 0) / np.count_nonzero(self.training_data.y)
            self.image_shape = self.training_data.y.shape[1:3]

        # Custom weight function for class weight balancing
        unweighted_bce = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)

        def weighted_bce(y_true, y_pred):
            weights = (y_true * (weighting - 1)) + 1
            bce = tf.expand_dims(unweighted_bce(y_true, y_pred), -1)
            return tf.reduce_mean(bce * weights)

        input_layer = tf.keras.layers.Input(shape=(self.image_shape[0], self.image_shape[1], 1))
        multires_unet = UNet.multi_res_unet(input_layer, output_channels=self.output_channels, conv_filters=self.filters)
        model = tf.keras.models.Model(input_layer, multires_unet)

        if isinstance(self.lr_decay, float):
            opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, decay=self.lr_decay)
        else:
            opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, decay=0.0)

        model.compile(loss=weighted_bce, optimizer=opt, metrics=['mae', 'acc'])
        return model

    ########################
    # Network Architecture #
    ########################
    @staticmethod
    def conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1), activation='relu', name=None):
        """
        2D Convolutional layers

        Arguments:
            x {keras layer} -- input layer
            filters {int} -- number of filters
            num_row {int} -- number of rows in filters
            num_col {int} -- number of columns in filters

        Keyword Arguments:
            padding {str} -- mode of padding (default: {'same'})
            strides {tuple} -- stride of convolution operation (default: {(1, 1)})
            activation {str} -- activation function (default: {'relu'})
            name {str} -- name of the layer (default: {None})

        Returns:
            [keras layer] -- [output layer]
        """
        x = tf.keras.layers.Conv2D(filters, (num_row, num_col), strides=strides, padding=padding, use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization(axis=3, scale=False)(x)
        if activation is None:
            return x
        x = tf.keras.layers.Activation(activation, name=name)(x)
        return x

    @staticmethod
    def trans_conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(2, 2), name=None):
        """
        2D Transposed Convolutional layers

        Arguments:
            x {keras layer} -- input layer
            filters {int} -- number of filters
            num_row {int} -- number of rows in filters
            num_col {int} -- number of columns in filters

        Keyword Arguments:
            padding {str} -- mode of padding (default: {'same'})
            strides {tuple} -- stride of convolution operation (default: {(2, 2)})
            name {str} -- name of the layer (default: {None})

        Returns:
            [keras layer] -- [output layer]
        """
        x = tf.keras.layers.Conv2DTranspose(filters, (num_row, num_col), strides=strides, padding=padding)(x)
        x = tf.keras.layers.BatchNormalization(axis=3, scale=False)(x)
        return x

    @staticmethod
    def multi_res_block(u, inp, alpha=1.67):
        """
        MultiRes Block

        Arguments:
            U {int} -- Number of filters in a corrsponding UNet stage
            inp {keras layer} -- input layer

        Returns:
            [keras layer] -- [output layer]
        """
        w = alpha * u
        shortcut = inp
        shortcut = UNet.conv2d_bn(shortcut, int(w*0.167) + int(w*0.333) + int(w*0.5), 1, 1, activation=None, padding='same')
        conv3x3 = UNet.conv2d_bn(inp, int(w*0.167), 3, 3, activation='relu', padding='same')
        conv5x5 = UNet.conv2d_bn(conv3x3, int(w*0.333), 3, 3, activation='relu', padding='same')
        conv7x7 = UNet.conv2d_bn(conv5x5, int(w*0.5), 3, 3, activation='relu', padding='same')
        out = tf.keras.layers.concatenate([conv3x3, conv5x5, conv7x7], axis=3)
        out = tf.keras.layers.BatchNormalization(axis=3)(out)
        out = tf.keras.layers.add([shortcut, out])
        out = tf.keras.layers.Activation('relu')(out)
        out = tf.keras.layers.BatchNormalization(axis=3)(out)
        return out

    @staticmethod
    def res_path(filters, length, inp):
        """
        ResPath

        Arguments:
            filters {int} -- [description]
            length {int} -- length of ResPath
            inp {keras layer} -- input layer

        Returns:
            [keras layer] -- [output layer]
        """
        shortcut = inp
        shortcut = UNet.conv2d_bn(shortcut, filters, 1, 1, activation=None, padding='same')
        out = UNet.conv2d_bn(inp, filters, 3, 3, activation='relu', padding='same')
        out = tf.keras.layers.add([shortcut, out])
        out = tf.keras.layers.Activation('relu')(out)
        out = tf.keras.layers.BatchNormalization(axis=3)(out)

        for i in range(length-1):
            shortcut = out
            shortcut = UNet.conv2d_bn(shortcut, filters, 1, 1, activation=None, padding='same')
            out = UNet.conv2d_bn(out, filters, 3, 3, activation='relu', padding='same')
            out = tf.keras.layers.add([shortcut, out])
            out = tf.keras.layers.Activation('relu')(out)
            out = tf.keras.layers.BatchNormalization(axis=3)(out)
        return out

    @staticmethod
    def multi_res_unet(inputs, output_channels=1, conv_filters=16):
        """
        MultiResUNet

        Arguments:
            inputs {tf.keras.layers.Input} -- Input Layer
            output_channels {int} -- number of channels in the output
            conv_filters {int} -- number of convolutional filters

        Returns:
            [keras model] -- MultiResUNet model
        """
        filters = conv_filters
        # Make sure the input has a size that can be downsampled at least 4 times (if not, use reflection padding)
        padding_height = ((2 ** 4 - inputs.shape[1] % 2 ** 4) % 2 ** 4)
        padding_width = ((2 ** 4 - inputs.shape[2] % 2 ** 4) % 2 ** 4)
        inputs_padded = ReflectionPadding2D((padding_width, padding_height))(inputs)

        mresblock1 = UNet.multi_res_block(filters, inputs_padded)
        pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(mresblock1)
        mresblock1 = UNet.res_path(filters, 4, mresblock1)

        mresblock2 = UNet.multi_res_block(filters * 2, pool1)
        pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(mresblock2)
        mresblock2 = UNet.res_path(filters * 2, 3, mresblock2)

        mresblock3 = UNet.multi_res_block(filters * 4, pool2)
        pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(mresblock3)
        mresblock3 = UNet.res_path(filters * 4, 2, mresblock3)

        mresblock4 = UNet.multi_res_block(filters * 8, pool3)
        pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(mresblock4)
        mresblock4 = UNet.res_path(filters * 8, 1, mresblock4)

        mresblock5 = UNet.multi_res_block(filters * 16, pool4)

        up6 = tf.keras.layers.concatenate([tf.keras.layers.Conv2DTranspose(filters*8, (2, 2), strides=(2, 2), padding='same')(mresblock5), mresblock4], axis=3)
        mresblock6 = UNet.multi_res_block(32 * 8, up6)

        up7 = tf.keras.layers.concatenate([tf.keras.layers.Conv2DTranspose(filters*4, (2, 2), strides=(2, 2), padding='same')(mresblock6), mresblock3], axis=3)
        mresblock7 = UNet.multi_res_block(32 * 4, up7)

        up8 = tf.keras.layers.concatenate([tf.keras.layers.Conv2DTranspose(filters*2, (2, 2), strides=(2, 2), padding='same')(mresblock7), mresblock2], axis=3)
        mresblock8 = UNet.multi_res_block(32 * 2, up8)

        up9 = tf.keras.layers.concatenate([tf.keras.layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(mresblock8), mresblock1], axis=3)
        mresblock9 = UNet.multi_res_block(filters, up9)

        cropped = tf.keras.layers.Cropping2D(cropping=((padding_height // 2, padding_height // 2 + padding_height % 2), (padding_width // 2, padding_width // 2 + padding_width % 2)))(mresblock9)

        if output_channels == 1:
            conv10 = UNet.conv2d_bn(cropped, 1, 1, 1, activation='sigmoid', name='output')
        else:
            conv10 = tf.keras.layers.Conv2D(output_channels, (1, 1), strides=1, activation=None)(cropped)
            conv10 = tf.keras.layers.Activation('softmax', name='output')(conv10)

        return conv10


class ReflectionPadding2D(tf.keras.layers.Layer):
    """Implements Reflection Padding as a layer.
    Args:
        padding(tuple): Total amount of padding for the spatial dimensions (will be split between left/right and up/down).
    Returns:
        A padded tensor with the same type as the input tensor.
    """

    def __init__(self, padding=(2, 2), **kwargs):
        self.padding = tuple(padding)
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def call(self, input_tensor, mask=None):
        padding_width, padding_height = self.padding
        padding_tensor = [
            [0, 0],
            [padding_height // 2, padding_height // 2 + padding_height % 2],
            [padding_width // 2, padding_width // 2 + padding_width % 2],
            [0, 0],
        ]
        return tf.pad(input_tensor, padding_tensor, mode="REFLECT")


if __name__ == '__main__':
    unet = UNet(root_dir='./', image_dir='images', mask_dir='masks')
