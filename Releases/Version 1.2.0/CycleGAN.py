import os
import time
from PIL import Image
from scipy import ndimage
import numpy as np
import keras
from tqdm import tqdm
import random

import HelperFunctions

if os.environ["KERAS_BACKEND"] == "torch":
    import torch
elif os.environ["KERAS_BACKEND"] == "tensorflow":
    import tensorflow as tf
else:
    raise NotImplementedError('Unsupported backend. Please make sure the environment variable "KERAS_BACKEND" is set either to "tensorflow" or to "torch".')


class CycleGAN:
    def __init__(self, root_dir='./', image_shape=(384, 384, 1), allow_memory_growth=True, use_gpus_no=(0, )):
        # Training and hyperparameters
        self.batch_size = 2
        self.epochs = 50
        self.learning_rate = 2e-4
        self.use_data_loader = False
        self.filters = 32
        self.num_downsampling_blocks_gen = 3
        self.num_residual_blocks_gen = 9
        self.num_upsampling_blocks_gen = 3
        self.num_downsampling_blocks_disc = 2

        # GPU configuration
        self.allow_memory_growth = allow_memory_growth
        self.use_gpus_no = use_gpus_no
        if os.environ["KERAS_BACKEND"] == "tensorflow":
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                try:
                    gpus_used = gpus  # Only the ones to be used are visible due to environment variable "CUDA_VISIBLE_DEVICES"
                    # gpus_used = [gpus[i] for i in self.use_gpus_no]
                    # tf.config.set_visible_devices(gpus_used, 'GPU')
                    if self.allow_memory_growth:
                        for gpu in gpus_used:
                            tf.config.experimental.set_memory_growth(gpu, True)
                    # logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                    # print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
                    # print(gpus_used)
                    if len(gpus) > 1:
                        distribution = keras.distribution.DataParallel(devices=gpus_used)
                        keras.distribution.set_distribution(distribution)
                except RuntimeError as e:
                    # Visible devices must be set before GPUs have been initialized
                    print(e)
        elif os.environ["KERAS_BACKEND"] == "torch" and torch.cuda.is_available() and len(self.use_gpus_no) > 1:
            distribution = keras.distribution.DataParallel(devices=(torch.cuda.device(i) for i in self.use_gpus_no))
            keras.distribution.set_distribution(distribution)

        # Losses and Loss Weights; cycle and identity loss for A->B (images->binary masks) can be set to BinaryCrossentropy instead of MeanAbsoluteError, but this makes the network asymmetric (last layer in Generator_A (i.e., the generator that generates B from A or Masks from Images) will be sigmoid instead of tanh), and it might be necessary to adjust lambda_cycle_A (and lambda_identity_A if used)
        self.lambda_cycle_a = 10
        self.lambda_cycle_b = 10
        self.use_binary_crossentropy = False

        # Linear decay of learning rate, for both discriminators and generators
        self.use_linear_decay = True
        self.decay_epoch = int(0.75 * self.epochs)  # The epoch where the linear decay of the learning rates start

        # Identity loss - send images from B to G_A2B (and the opposite) to teach identity mappings; set to value > 0 (e.g., 0.5) to enable identity mapping - not compatible with use_binary_crossentropy
        self.lambda_identity_a = 0.5
        self.lambda_identity_b = 0.5
        assert not (self.use_binary_crossentropy and (self.lambda_identity_a > 0 or self.lambda_identity_a > 0)), 'In the current implementation, binary crossentropy cannot be used with identity mapping. Please set either self.use_binary_crossentropy = False or both identity losses to 0.'

        # Skip Connection - adds a skip connection between the input and output in the generator (conceptually similar to an identity mapping)
        self.use_skip_connection = True

        # Resize convolution - instead of transpose convolution in deconvolution layers - can reduce checkerboard artifacts but the blurring might affect the cycle-consistency
        self.use_resize_convolution = False

        # Discriminator regularization - avoid "overtraining" the discriminator
        self.label_smoothing_factor = 0.0  # Label smoothing factor - set to a small value (e.g., 0.1) to avoid overconfident discriminator guesses and very low discriminator losses (too strong discriminators can be problematic for generators due to adversarial nature of GANs)
        self.gaussian_noise_value = 0.15   # Set to a small value (e.g., 0.15) to add Gaussian Noise to the discriminator layers (can help against mode collapse and "overtraining" the discriminator)

        # Invert images
        self.invert_images = False  # Set to true to invert images (can help if masks are white on black background, but particles are black on light background (e.g. in TEM or TSEM images)

        # Image pool size for discriminator
        self.image_pool_size = 50

        # Set up directories and variables
        self.gen_a = None
        self.gen_b = None
        self.disc_a = None
        self.disc_b = None
        self.adv_loss_fn = None
        self.model = None
        self.data = None
        self.kernel_init = None
        self.gamma_init = None
        self.root_dir = root_dir
        self.model_dir = os.path.join(self.root_dir, '2_CycleGAN', 'Models')
        self.image_shape = image_shape
        self.prefix = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
        self.cycle_loss_fn_a = keras.losses.MeanAbsoluteError()
        self.cycle_loss_fn_b = keras.losses.MeanAbsoluteError()
        self.identity_loss_fn_a = keras.losses.MeanAbsoluteError()
        self.identity_loss_fn_b = keras.losses.MeanAbsoluteError()
        self.image_pool_a = ImagePool(batch_size=self.batch_size, pool_size=self.image_pool_size)
        self.image_pool_b = ImagePool(batch_size=self.batch_size, pool_size=self.image_pool_size)

        # Arrays for training and test data
        self.train_a = HelperFunctions.get_image_file_paths_from_directory(os.path.join(self.root_dir, '2_CycleGAN', 'data', 'trainA'))
        self.test_a = HelperFunctions.get_image_file_paths_from_directory(os.path.join(self.root_dir, '2_CycleGAN', 'data', 'testA'))
        self.train_b = HelperFunctions.get_image_file_paths_from_directory(os.path.join(self.root_dir, '2_CycleGAN', 'data', 'trainB'))
        self.test_b = HelperFunctions.get_image_file_paths_from_directory(os.path.join(self.root_dir, '2_CycleGAN', 'data', 'testB'))

    def create_model(self):
        if self.use_binary_crossentropy:
            self.cycle_loss_fn_a = keras.losses.BinaryCrossentropy()
            self.cycle_loss_fn_b = keras.losses.MeanAbsoluteError()
            self.identity_loss_fn_a = keras.losses.BinaryCrossentropy()
            self.identity_loss_fn_b = keras.losses.MeanAbsoluteError()

        # Weights initializer for the layers.
        # keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
        self.kernel_init = keras.initializers.GlorotUniform(keras.random.SeedGenerator(0))
        # Gamma initializer for instance normalization.
        # self.gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
        self.gamma_init = 'ones'
        # Loss function for evaluating adversarial loss
        self.adv_loss_fn = keras.losses.MeanSquaredError()

        # Build the generators and discriminators
        self.gen_a = self.get_resnet_generator(name="generator_A",
                                               filters=self.filters,
                                               num_downsampling_blocks=self.num_downsampling_blocks_gen,
                                               num_residual_blocks=self.num_residual_blocks_gen,
                                               num_upsample_blocks=self.num_upsampling_blocks_gen,
                                               use_binary_crossentropy=self.use_binary_crossentropy,
                                               )
        self.gen_b = self.get_resnet_generator(name="generator_B",
                                               filters=self.filters,
                                               num_downsampling_blocks=self.num_downsampling_blocks_gen,
                                               num_residual_blocks=self.num_residual_blocks_gen,
                                               num_upsample_blocks=self.num_upsampling_blocks_gen,
                                               use_binary_crossentropy=False,
                                               )

        padding = "valid"  # The original implementation used padding="same", however, this leads to discrepancies when using pytorch as a backend.
        self.disc_a = self.get_discriminator(name="discriminator_A", num_downsampling_blocks=self.num_downsampling_blocks_disc, filters=2 * self.filters, padding=padding)
        self.disc_b = self.get_discriminator(name="discriminator_B", num_downsampling_blocks=self.num_downsampling_blocks_disc, filters=2 * self.filters, padding=padding)

        # Create CycleGAN model
        model = CycleGanModel(
            generator_a=self.gen_a,
            generator_b=self.gen_b,
            discriminator_a=self.disc_a,
            discriminator_b=self.disc_b,
            image_pool_a=self.image_pool_a,
            image_pool_b=self.image_pool_b,
            lambda_cycle_a=self.lambda_cycle_a,
            lambda_cycle_b=self.lambda_cycle_b,
            lambda_identity_a=self.lambda_identity_a,
            lambda_identity_b=self.lambda_identity_b,
        )

        # Compile the model
        model.compile(
            gen_a_optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=0.5),
            gen_b_optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=0.5),
            disc_x_optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=0.5),
            disc_y_optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=0.5),
            gen_loss_fn=self.generator_loss_fn,
            disc_loss_fn=self.discriminator_loss_fn,
            cycle_loss_fn_a=self.cycle_loss_fn_a,
            cycle_loss_fn_b=self.cycle_loss_fn_b,
            identity_loss_fn_a=self.identity_loss_fn_a,
            identity_loss_fn_b=self.identity_loss_fn_b,
        )

        return model

    def start_training(self):
        if not os.path.isdir(os.path.join(self.model_dir, self.prefix)):
            os.mkdir(os.path.join(self.model_dir, self.prefix))
        if not os.path.isdir(os.path.join(self.root_dir, '2_CycleGAN', 'images', self.prefix)):
            os.mkdir(os.path.join(self.root_dir, '2_CycleGAN', 'images', self.prefix))

        self.decay_epoch = int(0.75 * self.epochs)

        self.test_a = self.load_images(self.test_a, scale_for_binary_crossentropy=False, invert=self.invert_images)
        self.test_b = self.load_images(self.test_b, scale_for_binary_crossentropy=self.use_binary_crossentropy)
        if not self.use_data_loader:
            self.train_a = self.load_images(self.train_a, scale_for_binary_crossentropy=False, invert=self.invert_images)
            self.train_b = self.load_images(self.train_b, scale_for_binary_crossentropy=self.use_binary_crossentropy)

        self.data = DataLoader(self.train_a, self.train_b, batch_size=self.batch_size, use_dataloader=self.use_data_loader, scale_for_binary_crossentropy=self.use_binary_crossentropy, invert_images=self.invert_images)

        # Create the model
        self.model = self.create_model()

        # Callbacks
        plotter = GANMonitor(self.test_a, self.test_b, output_dir=os.path.join(self.root_dir, '2_CycleGAN', 'images', self.prefix), num_img=2)
        checkpoint_filepath = os.path.join(self.model_dir, self.prefix, 'checkpoints_{epoch:03d}.keras')
        model_checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath)
        csv_logger = keras.callbacks.CSVLogger(os.path.join(self.model_dir, self.prefix, 'training_log.csv'), separator=';', append=True)
        callbacks_list = [plotter,
                          model_checkpoint_callback,
                          csv_logger]
        if self.use_linear_decay:
            rate_scheduler = keras.callbacks.LearningRateScheduler(self.linear_decay)
            callbacks_list.append(rate_scheduler)

        # Train the model
        self.model.fit(
            self.data,
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=callbacks_list,
        )

        self.model.save(os.path.join(self.model_dir,  self.prefix, 'model.keras'))
        return self.model

    def run_inference(self, files, output_directory, source_domain, model=None, tile_images=False, min_overlap=2, manage_overlap_mode=2, use_gpu=False):
        if self.model is None:
            if model is None:
                # Load the most recent model
                self.model = keras.models.load_model(os.path.join(self.model_dir, os.listdir(self.model_dir)[-1], 'model.keras'))
            elif isinstance(model, str):
                # Load the specified model
                self.model = keras.models.load_model(model)
            else:
                self.model = model

        if 'a' in source_domain.lower():
            generator_model = self.model.gen_a
        else:
            generator_model = self.model.gen_b

        input_files = HelperFunctions.load_and_preprocess_images(files, normalization_range=(-1, 1))
        file_names = HelperFunctions.get_image_file_paths_from_directory(files)

        if not tile_images and input_files[0].shape != self.image_shape:
            self.image_shape = input_files[0].shape
            model_new = self.create_model()
            if 'a' in source_domain.lower():
                generator_model_new = model_new.gen_a
            else:
                generator_model_new = model_new.gen_b
            generator_model_new.set_weights(generator_model.get_weights())
            generator_model = generator_model_new

        if os.environ["KERAS_BACKEND"] == 'torch':
            if use_gpu:
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')
            generator_model.to(device)

        if use_gpu:
            device = 'gpu:0'
        else:
            device = 'cpu'

        with keras.device(device):
            for i in tqdm(range(0, input_files.shape[0])):
                input_file = input_files[i]
                if 'a' in source_domain.lower() and self.invert_images:
                    input_file *= -1
                if tile_images:
                    tiles = np.asarray(HelperFunctions.tile_image(input_file, self.image_shape[0], self.image_shape[1], min_overlap=min_overlap))
                    # prediction = np.asarray([CycleGanModel.to_numpy_array(self._generate_image(generator_model, img.reshape(1, img.shape[0], img.shape[1], img.shape[2]), device))[0] for img in tiles])
                    prediction = np.asarray([CycleGanModel.to_numpy_array(generator_model(img.reshape(1, img.shape[0], img.shape[1], img.shape[2]), training=False))[0] for img in tiles])
                    img = HelperFunctions.stitch_image(prediction, input_file.shape[1], input_file.shape[0], min_overlap=min_overlap, manage_overlap_mode=manage_overlap_mode)
                else:
                    # prediction = CycleGanModel.to_numpy_array(self._generate_image(generator_model, input_file.reshape(1, input_file.shape[0], input_file.shape[1], input_file.shape[2]), device))
                    prediction = CycleGanModel.to_numpy_array(generator_model(input_file.reshape(1, input_file.shape[0], input_file.shape[1], input_file.shape[2]), training=False))
                    img = prediction[0].copy()
                img = img[:, :, 0]
                if 'b' in source_domain.lower() and self.invert_images:
                    img *= -1
                img -= np.min(img)
                img /= np.max(img)
                img *= 255
                img = img.astype(np.uint8)
                Image.fromarray(img).save(os.path.join(output_directory, os.path.split(file_names[i])[-1]))

    @staticmethod
    def load_images(image_list, scale_for_binary_crossentropy=False, invert=False):
        if scale_for_binary_crossentropy:
            r = (0, 1)
        else:
            r = (-1, 1)
        images = HelperFunctions.load_and_preprocess_images(input_dir_or_filelist=image_list, threshold_value=None, normalization_range=r, output_channels=1, contrast_optimization_range=None)

        if invert:
            images *= -1.0

        return images

    def generator_loss_fn(self, fake):
        fake_loss = self.adv_loss_fn(keras.ops.ones_like(fake) * (1.0-self.label_smoothing_factor) + (self.label_smoothing_factor/2), fake)
        return fake_loss

    def discriminator_loss_fn(self, real, fake):
        real_loss = self.adv_loss_fn(keras.ops.ones_like(real) * (1.0-self.label_smoothing_factor) + (self.label_smoothing_factor/2), real)
        fake_loss = self.adv_loss_fn(keras.ops.zeros_like(fake) * (1.0-self.label_smoothing_factor) + (self.label_smoothing_factor/2), fake)
        return (real_loss + fake_loss) * 0.5, real_loss, fake_loss

    def linear_decay(self, epoch, current_lr):
        if epoch < self.decay_epoch:
            return self.learning_rate
        power = 1  # 1 -> Linear Decay
        initial_lrate = self.learning_rate
        decay = (1 - ((epoch-self.decay_epoch) / float(self.epochs-self.decay_epoch))) ** power
        lrate = initial_lrate * decay
        return lrate

    ########################
    # Network Architecture #
    ########################

    def residual_block(self, input_tensor, activation, kernel_size=(3, 3), strides=(1, 1), padding="valid", use_bias=False):
        dim = input_tensor.shape[-1]

        x = ReflectionPadding2D()(input_tensor)
        x = keras.layers.Conv2D(dim, kernel_size, strides=strides, kernel_initializer=self.kernel_init, padding=padding, use_bias=use_bias)(x)
        # x = tfa.layers.InstanceNormalization(axis=3, center=True, gamma_initializer=self.gamma_init, epsilon=1e-5)(x, training=True)
        x = keras.layers.GroupNormalization(axis=3, groups=-1, center=True, gamma_initializer=self.gamma_init, epsilon=1e-5)(x, training=True)
        x = activation(x)

        x = ReflectionPadding2D()(x)
        x = keras.layers.Conv2D(dim, kernel_size, strides=strides, kernel_initializer=self.kernel_init, padding=padding, use_bias=use_bias)(x)
        # x = tfa.layers.InstanceNormalization(axis=3, center=True, gamma_initializer=self.gamma_init, epsilon=1e-5)(x, training=True)
        x = keras.layers.GroupNormalization(axis=3, groups=-1, center=True, gamma_initializer=self.gamma_init, epsilon=1e-5)(x, training=True)
        x = keras.layers.add([input_tensor, x])
        return x

    def downsample(self, x, filters, activation, kernel_size=(3, 3), strides=(2, 2), padding="same", use_bias=False):
        x = keras.layers.Conv2D(filters, kernel_size, strides=strides, kernel_initializer=self.kernel_init, padding=padding, use_bias=use_bias)(x)
        # x = tfa.layers.InstanceNormalization(axis=3, center=True, gamma_initializer=self.gamma_init, epsilon=1e-5)(x, training=True)
        x = keras.layers.GroupNormalization(axis=3, groups=-1, center=True, gamma_initializer=self.gamma_init, epsilon=1e-5)(x, training=True)
        if activation:
            x = activation(x)
        return x

    def upsample(self, x, filters, activation, kernel_size=(3, 3), strides=(2, 2), padding="same", use_bias=False):
        if self.use_resize_convolution:
            x = keras.layers.UpSampling2D(size=(2, 2))(x)  # Nearest neighbor upsampling
            x = ReflectionPadding2D()(x)
            x = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=1, padding='valid', use_bias=use_bias)(x)
        else:
            x = keras.layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding, kernel_initializer=self.kernel_init, use_bias=use_bias)(x)
        # x = tfa.layers.InstanceNormalization(axis=3, center=True, gamma_initializer=self.gamma_init, epsilon=1e-5)(x, training=True)
        x = keras.layers.GroupNormalization(axis=3, groups=-1, center=True, gamma_initializer=self.gamma_init, epsilon=1e-5)(x, training=True)
        if activation:
            x = activation(x)
        return x

    def get_resnet_generator(self, filters=64, num_downsampling_blocks=2, num_residual_blocks=9, num_upsample_blocks=2, name=None, use_binary_crossentropy=False, padding="same"):
        # Input
        img_input = keras.layers.Input(shape=self.image_shape, name=name + "_img_input")

        # Make sure the input has a size that can be downsampled at least num_downsampling_blocks times (if not, use reflection padding)
        padding_height = ((2 ** num_downsampling_blocks - img_input.shape[1] % 2 ** num_downsampling_blocks) % 2 ** num_downsampling_blocks)
        padding_width = ((2 ** num_downsampling_blocks - img_input.shape[2] % 2 ** num_downsampling_blocks) % 2 ** num_downsampling_blocks)
        x = ReflectionPadding2D(padding=(padding_width, padding_height))(img_input)
        x = ReflectionPadding2D(padding=(6, 6))(x)

        # x = ReflectionPadding2D(padding=(6, 6))(img_input)
        # 7 x 7 Convolution
        x = keras.layers.Conv2D(filters, (7, 7), kernel_initializer=self.kernel_init, use_bias=False)(x)
        # x = tfa.layers.InstanceNormalization(axis=3, center=True, gamma_initializer=self.gamma_init, epsilon=1e-5)(x, training=True)
        x = keras.layers.GroupNormalization(axis=3, groups=-1, center=True, gamma_initializer=self.gamma_init, epsilon=1e-5)(x, training=True)
        x = keras.layers.Activation("relu")(x)

        # Downsampling
        for _ in range(num_downsampling_blocks):
            filters *= 2
            x = self.downsample(x, filters=filters, activation=keras.layers.Activation("relu"), padding=padding)

        # Residual blocks
        for _ in range(num_residual_blocks):
            x = self.residual_block(x, activation=keras.layers.Activation("relu"))

        # Upsampling
        for _ in range(num_upsample_blocks):
            filters //= 2
            x = self.upsample(x, filters, activation=keras.layers.Activation("relu"), padding=padding)

        # Final block
        x = ReflectionPadding2D(padding=(6, 6))(x)
        x = keras.layers.Conv2D(self.image_shape[-1], (7, 7), padding="valid")(x)
        # x = keras.layers.Cropping2D(cropping=((padding_height // 2, padding_height // 2 + padding_height % 2), (padding_width // 2, padding_width // 2 + padding_width % 2)))(x)

        if self.use_skip_connection:
            shortcut = img_input
            shortcut = keras.layers.Conv2D(filters, (1, 1), kernel_initializer=self.kernel_init, activation=None, use_bias=False)(shortcut)
            # shortcut = tfa.layers.InstanceNormalization(axis=3, center=True, gamma_initializer=self.gamma_init, epsilon=1e-5)(shortcut, training=True)
            shortcut = keras.layers.GroupNormalization(axis=3, groups=-1, center=True, gamma_initializer=self.gamma_init, epsilon=1e-5)(shortcut, training=True)
            shortcut = keras.layers.Activation("relu")(shortcut)

            out = ReflectionPadding2D()(img_input)
            out = keras.layers.Conv2D(filters, (3, 3), kernel_initializer=self.kernel_init, use_bias=False)(out)
            # out = tfa.layers.InstanceNormalization(axis=3, center=True, gamma_initializer=self.gamma_init, epsilon=1e-5)(out, training=True)
            out = keras.layers.GroupNormalization(axis=3, groups=-1, center=True, gamma_initializer=self.gamma_init, epsilon=1e-5)(out, training=True)
            out = keras.layers.Activation("relu")(out)

            out = keras.layers.add([shortcut, out])
            # out = tfa.layers.InstanceNormalization(axis=3, center=True, gamma_initializer=self.gamma_init, epsilon=1e-5)(out, training=True)
            out = keras.layers.GroupNormalization(axis=3, groups=-1, center=True, gamma_initializer=self.gamma_init, epsilon=1e-5)(out, training=True)
            out = keras.layers.Activation("relu")(out)

            x = keras.layers.concatenate([out, x], axis=3)
            x = keras.layers.Conv2D(self.image_shape[-1], (1, 1), kernel_initializer=self.kernel_init, activation=None, use_bias=False)(x)

        if use_binary_crossentropy:
            x = keras.layers.Activation("sigmoid", name=f'output_{name}')(x)
        else:
            x = keras.layers.Activation("tanh", name=f'output_{name}')(x)

        model = keras.models.Model(img_input, x, name=name)
        return model

    def get_discriminator(self, filters=32, num_downsampling_blocks=3, name=None, padding="same"):
        img_input = keras.layers.Input(shape=self.image_shape, name=name + "_img_input")
        if self.gaussian_noise_value > 0:
            x = keras.layers.GaussianNoise(self.gaussian_noise_value)(img_input)
            x = keras.layers.Conv2D(filters, (4, 4), strides=(2, 2), padding=padding, kernel_initializer=self.kernel_init)(x)
        else:
            x = keras.layers.Conv2D(filters, (4, 4), strides=(2, 2), padding=padding, kernel_initializer=self.kernel_init)(img_input)

        x = keras.layers.LeakyReLU(0.2)(x)

        for num_downsample_block in range(num_downsampling_blocks):
            filters *= 2
            if num_downsample_block < 3:
                if self.gaussian_noise_value > 0:
                    x = keras.layers.GaussianNoise(self.gaussian_noise_value)(x)
                x = self.downsample(x, filters=filters, activation=keras.layers.LeakyReLU(0.2), kernel_size=(4, 4), strides=(2, 2), padding=padding)
            else:
                if self.gaussian_noise_value > 0:
                    x = keras.layers.GaussianNoise(self.gaussian_noise_value)(x)
                x = self.downsample(x, filters=filters, activation=keras.layers.LeakyReLU(0.2), kernel_size=(4, 4), strides=(1, 1), padding=padding)

        if self.gaussian_noise_value > 0:
            x = keras.layers.GaussianNoise(self.gaussian_noise_value)(x)
        x = keras.layers.Conv2D(1, (4, 4), strides=(1, 1), padding=padding, kernel_initializer=self.kernel_init, name=f'output_{name}')(x)

        model = keras.models.Model(inputs=img_input, outputs=x, name=name)
        return model


class DataLoader(keras.utils.Sequence):
    def __init__(self, train_a, train_b, batch_size=1, use_dataloader=False, scale_for_binary_crossentropy=False, invert_images=False, **kwargs):
        super(DataLoader, self).__init__(**kwargs)
        self.batch_size = batch_size
        self.train_a = train_a
        self.train_b = train_b
        self.use_dataloader = use_dataloader
        self.scale_for_binary_crossentropy = scale_for_binary_crossentropy
        self.invert_images = invert_images

    def __len__(self):
        return int(min(len(self.train_a), len(self.train_b)) / float(self.batch_size))

    def __getitem__(self, idx):
        real_images_a = self.train_a[idx * self.batch_size:(idx + 1) * self.batch_size]
        real_images_b = self.train_b[idx * self.batch_size:(idx + 1) * self.batch_size]

        if self.use_dataloader:
            real_images_a = CycleGAN.load_images(real_images_a, False, invert=self.invert_images)
            real_images_b = CycleGAN.load_images(real_images_b, self.scale_for_binary_crossentropy)

        return np.asarray(real_images_a), np.asarray(real_images_b)
        
    def on_epoch_end(self):
        np.random.shuffle(self.train_a)
        np.random.shuffle(self.train_b)


@keras.saving.register_keras_serializable()
class ReflectionPadding2D(keras.layers.Layer):
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
        if padding_width == 0 and padding_height == 0:
            return input_tensor

        padding_tensor = (
            (0, 0),
            (padding_height // 2, padding_height // 2 + padding_height % 2),
            (padding_width // 2, padding_width // 2 + padding_width % 2),
            (0, 0),
        )
        return keras.ops.pad(input_tensor, padding_tensor, mode="reflect")

    # def compute_output_shape(self, input_shape):
    #     return input_shape[0], input_shape[1] + self.padding[0], input_shape[2] + self.padding[1], input_shape[3]


@keras.saving.register_keras_serializable()
class CycleGanModel(keras.Model):
    def __init__(self, generator_a, generator_b, discriminator_a, discriminator_b, image_pool_a=None, image_pool_b=None, lambda_cycle_a=10.0, lambda_cycle_b=10.0, lambda_identity_a=0.5, lambda_identity_b=0.5, **kwargs):
        super(CycleGanModel, self).__init__(**kwargs)
        self.gen_a = generator_a
        self.gen_b = generator_b
        self.disc_a = discriminator_a
        self.disc_b = discriminator_b
        self.lambda_cycle_a = lambda_cycle_a
        self.lambda_cycle_b = lambda_cycle_b
        self.lambda_identity_a = lambda_identity_a
        self.lambda_identity_b = lambda_identity_b

        self.use_identity_loss = lambda_identity_a > 0 or lambda_identity_b > 0

        self.gen_a_optimizer = None
        self.gen_b_optimizer = None
        self.disc_a_optimizer = None
        self.disc_b_optimizer = None
        self.generator_loss_fn = None
        self.discriminator_loss_fn = None
        self.cycle_loss_fn_a = None
        self.cycle_loss_fn_b = None
        self.identity_loss_fn_a = None
        self.identity_loss_fn_b = None

        self.image_pool_a = image_pool_a
        self.image_pool_b = image_pool_b

        if self.image_pool_a is None:
            self.image_pool_a = ImagePool(1, 0)
        if self.image_pool_b is None:
            self.image_pool_b = ImagePool(1, 0)

        self.losses_log = {'d_a': [], 'd_b': [], 'd_fake_a': [], 'd_real_a': [], 'd_fake_b': [], 'd_real_b': [], 'g_a': [], 'g_b': [], 'g_adv_a': [], 'g_adv_b': [], 'g_cyc_a': [], 'g_cyc_b': [], 'g_id_a': [], 'g_id_b': []}
        self.d_a_loss_tracker = keras.metrics.Mean(name="d_a")
        self.d_b_loss_tracker = keras.metrics.Mean(name="d_b")
        self.d_fake_a_loss_tracker = keras.metrics.Mean(name="d_fake_a")
        self.d_fake_b_loss_tracker = keras.metrics.Mean(name="d_fake_b")
        self.d_real_a_loss_tracker = keras.metrics.Mean(name="d_real_a")
        self.d_real_b_loss_tracker = keras.metrics.Mean(name="d_real_b")
        self.g_a_loss_tracker = keras.metrics.Mean(name="g_a")
        self.g_b_loss_tracker = keras.metrics.Mean(name="g_b")
        self.g_adv_a_loss_tracker = keras.metrics.Mean(name="g_adv_a")
        self.g_adv_b_loss_tracker = keras.metrics.Mean(name="g_adv_b")
        self.g_cyc_a_loss_tracker = keras.metrics.Mean(name="g_cyc_a")
        self.g_cyc_b_loss_tracker = keras.metrics.Mean(name="g_cyc_b")
        self.g_id_a_loss_tracker = keras.metrics.Mean(name="g_id_a")
        self.g_id_b_loss_tracker = keras.metrics.Mean(name="g_id_b")

        self.built = True

    @property
    def metrics(self):
        return [self.d_a_loss_tracker, self.d_b_loss_tracker, self.d_fake_a_loss_tracker, self.d_fake_b_loss_tracker, self.d_real_a_loss_tracker, self.d_real_b_loss_tracker, self.g_a_loss_tracker, self.g_b_loss_tracker, self.g_adv_a_loss_tracker, self.g_adv_b_loss_tracker, self.g_cyc_a_loss_tracker, self.g_cyc_b_loss_tracker, self.g_id_a_loss_tracker, self.g_id_b_loss_tracker]

    def compile(self, gen_a_optimizer, gen_b_optimizer, disc_x_optimizer, disc_y_optimizer, disc_loss_fn, gen_loss_fn, cycle_loss_fn_a=keras.losses.MeanAbsoluteError(), cycle_loss_fn_b=keras.losses.MeanAbsoluteError(), identity_loss_fn_a=keras.losses.MeanAbsoluteError(), identity_loss_fn_b=keras.losses.MeanAbsoluteError(), **kwargs):
        super(CycleGanModel, self).compile(**kwargs)
        self.gen_a_optimizer = gen_a_optimizer
        self.gen_b_optimizer = gen_b_optimizer
        self.disc_a_optimizer = disc_x_optimizer
        self.disc_b_optimizer = disc_y_optimizer
        self.generator_loss_fn = gen_loss_fn
        self.discriminator_loss_fn = disc_loss_fn
        self.cycle_loss_fn_a = cycle_loss_fn_a
        self.cycle_loss_fn_b = cycle_loss_fn_b
        self.identity_loss_fn_a = identity_loss_fn_a
        self.identity_loss_fn_b = identity_loss_fn_b

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                'gen_a': self.gen_a,
                'gen_b': self.gen_b,
                'disc_a': self.disc_a,
                'disc_b': self.disc_b,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        # Hack for manually deserializing nested objects (in this case keras.initializers.GlorotUniform(seed=keras.random.SeedGenerator(0)))
        for net in ('gen_a', 'gen_b', 'disc_a', 'disc_b'):
            for i in range(0, len(config[net]['config']['layers'])):
                if 'kernel_initializer' in config[net]['config']['layers'][i]['config'].keys():
                    config[net]['config']['layers'][i]['config']['kernel_initializer']['config']['seed'] = keras.saving.deserialize_keras_object(config[net]['config']['layers'][i]['config']['kernel_initializer']['config']['seed'])

        generator_a = keras.saving.deserialize_keras_object(config.pop('gen_a'))
        generator_b = keras.saving.deserialize_keras_object(config.pop('gen_b'))
        discriminator_a = keras.saving.deserialize_keras_object(config.pop('disc_a'))
        discriminator_b = keras.saving.deserialize_keras_object(config.pop('disc_b'))
        return cls(generator_a=generator_a, generator_b=generator_b, discriminator_a=discriminator_a, discriminator_b=discriminator_b, **config)

    def train_step(self, batch_data):
        if os.environ["KERAS_BACKEND"] == "torch":
            return self.train_step_torch(batch_data)
        elif os.environ["KERAS_BACKEND"] == "tensorflow":
            return self.train_step_tensorflow(batch_data)
        else:
            raise NotImplementedError('Unsupported backend. Please make sure the environment variable "KERAS_BACKEND" is set either to "tensorflow" or to "torch".')

    def train_step_torch(self, batch_data):
        # a = EM Images, b = Masks
        real_a, real_b = batch_data

        # Train the Generators
        # a to fake b
        fake_b = self.gen_a(real_a, training=True)
        # b to fake a
        fake_a = self.gen_b(real_b, training=True)

        # Cycle (x to fake y to fake x): x -> y -> x
        cycled_a = self.gen_b(fake_b, training=True)
        # Cycle (y to fake x to fake y) y -> x -> y
        cycled_b = self.gen_a(fake_a, training=True)

        # Identity mapping
        if self.use_identity_loss:
            same_a = self.gen_b(real_a, training=True)
            same_b = self.gen_a(real_b, training=True)

        # Discriminator output
        disc_fake_a = self.disc_a(fake_a, training=True)
        disc_fake_b = self.disc_b(fake_b, training=True)

        # Generator adversarial loss
        adv_loss_a = self.generator_loss_fn(disc_fake_b)
        adv_loss_b = self.generator_loss_fn(disc_fake_a)

        # Generator cycle loss
        cycle_loss_a = self.cycle_loss_fn_a(real_b, cycled_b) * self.lambda_cycle_a
        cycle_loss_b = self.cycle_loss_fn_b(real_a, cycled_a) * self.lambda_cycle_b

        # Generator identity loss
        if self.use_identity_loss:
            id_loss_a = (self.identity_loss_fn_a(real_b, same_b) * self.lambda_cycle_a * self.lambda_identity_a)
            id_loss_b = (self.identity_loss_fn_b(real_a, same_a) * self.lambda_cycle_b * self.lambda_identity_b)
        else:
            id_loss_a = 0
            id_loss_b = 0

        # Total generator loss
        total_gen_loss_a = adv_loss_a + cycle_loss_a + id_loss_a
        total_gen_loss_b = adv_loss_b + cycle_loss_b + id_loss_b

        # Zero the gradients
        self.gen_a.zero_grad()
        self.gen_b.zero_grad()

        # Get the gradients for the generators and update the weights
        total_gen_loss_a.backward(retain_graph=True)
        total_gen_loss_b.backward(retain_graph=True)

        with torch.no_grad():
            self.gen_a_optimizer.apply([v.value.grad for v in self.gen_a.trainable_weights], self.gen_a.trainable_weights)
            self.gen_b_optimizer.apply([v.value.grad for v in self.gen_b.trainable_weights], self.gen_b.trainable_weights)

        # Train the Discriminators
        disc_real_a = self.disc_a(real_a, training=True)
        disc_fake_a = self.disc_a(self.image_pool_a.query(fake_a.detach().clone()), training=True)

        disc_real_b = self.disc_b(real_b, training=True)
        disc_fake_b = self.disc_b(self.image_pool_b.query(fake_b.detach().clone()), training=True)

        # Discriminator loss
        total_disc_a_loss, disc_a_loss_real, disc_a_loss_fake = self.discriminator_loss_fn(disc_real_a, disc_fake_a)
        total_disc_b_loss, disc_b_loss_real, disc_b_loss_fake = self.discriminator_loss_fn(disc_real_b, disc_fake_b)

        # Zero the gradients
        self.disc_a.zero_grad()
        self.disc_b.zero_grad()

        # Get the gradients for the discriminators and update the weights
        total_disc_a_loss.backward()
        total_disc_b_loss.backward()
        with torch.no_grad():
            self.disc_a_optimizer.apply([v.value.grad for v in self.disc_a.trainable_weights], self.disc_a.trainable_weights)
        with torch.no_grad():
            self.disc_b_optimizer.apply([v.value.grad for v in self.disc_b.trainable_weights], self.disc_b.trainable_weights)

        # Update metrics and return their value.
        self.d_a_loss_tracker.update_state(total_disc_a_loss)
        self.d_b_loss_tracker.update_state(total_disc_b_loss)
        self.d_fake_a_loss_tracker.update_state(disc_a_loss_fake)
        self.d_fake_b_loss_tracker.update_state(disc_b_loss_fake)
        self.d_real_a_loss_tracker.update_state(disc_a_loss_real)
        self.d_real_b_loss_tracker.update_state(disc_b_loss_real)
        self.g_a_loss_tracker.update_state(total_gen_loss_a)
        self.g_b_loss_tracker.update_state(total_gen_loss_b)
        self.g_adv_a_loss_tracker.update_state(adv_loss_a)
        self.g_adv_b_loss_tracker.update_state(adv_loss_b)
        self.g_cyc_a_loss_tracker.update_state(cycle_loss_a)
        self.g_cyc_b_loss_tracker.update_state(cycle_loss_b)
        self.g_id_a_loss_tracker.update_state(id_loss_a)
        self.g_id_b_loss_tracker.update_state(id_loss_b)

        return {m.name: m.result() for m in self.metrics}

    def train_step_tensorflow(self, batch_data):
        # x = EM Images, y = Masks
        real_a, real_b = batch_data

        # Train the Generators
        with tf.GradientTape(persistent=True) as tape:
            # a = EM Images, b = Masks
            real_a, real_b = batch_data

            # Train the Generators
            # a to fake b
            fake_b = self.gen_a(real_a, training=True)
            # b to fake a
            fake_a = self.gen_b(real_b, training=True)

            # Cycle (x to fake y to fake x): x -> y -> x
            cycled_a = self.gen_b(fake_b, training=True)
            # Cycle (y to fake x to fake y) y -> x -> y
            cycled_b = self.gen_a(fake_a, training=True)

            # Identity mapping
            if self.use_identity_loss:
                same_a = self.gen_b(real_a, training=True)
                same_b = self.gen_a(real_b, training=True)

            # Discriminator output
            disc_fake_a = self.disc_a(fake_a, training=True)
            disc_fake_b = self.disc_b(fake_b, training=True)

            # Generator adversarial loss
            adv_loss_a = self.generator_loss_fn(disc_fake_b)
            adv_loss_b = self.generator_loss_fn(disc_fake_a)

            # Generator cycle loss
            cycle_loss_a = self.cycle_loss_fn_a(real_b, cycled_b) * self.lambda_cycle_a
            cycle_loss_b = self.cycle_loss_fn_b(real_a, cycled_a) * self.lambda_cycle_b

            # Generator identity loss
            if self.use_identity_loss:
                id_loss_a = (self.identity_loss_fn_a(real_b, same_b) * self.lambda_cycle_a * self.lambda_identity_a)
                id_loss_b = (self.identity_loss_fn_b(real_a, same_a) * self.lambda_cycle_b * self.lambda_identity_b)
            else:
                id_loss_a = 0
                id_loss_b = 0

            # Total generator loss
            total_gen_loss_a = adv_loss_a + cycle_loss_a + id_loss_a
            total_gen_loss_b = adv_loss_b + cycle_loss_b + id_loss_b

        # Get the gradients for the generators and update the weights
        self.gen_a_optimizer.apply(tape.gradient(total_gen_loss_a, self.gen_a.trainable_variables), self.gen_a.trainable_variables)
        self.gen_b_optimizer.apply(tape.gradient(total_gen_loss_b, self.gen_b.trainable_variables), self.gen_b.trainable_variables)

        # Train the Discriminators
        with tf.GradientTape(persistent=True) as tape:
            disc_real_a = self.disc_a(real_a, training=True)
            disc_fake_a = self.disc_a(self.image_pool_a.query(keras.ops.copy(tf.stop_gradient(fake_a))), training=True)

            disc_real_b = self.disc_b(real_b, training=True)
            disc_fake_b = self.disc_b(self.image_pool_b.query(keras.ops.copy(tf.stop_gradient(fake_b))), training=True)

            # Discriminator loss
            total_disc_a_loss, disc_a_loss_real, disc_a_loss_fake = self.discriminator_loss_fn(disc_real_a, disc_fake_a)
            total_disc_b_loss, disc_b_loss_real, disc_b_loss_fake = self.discriminator_loss_fn(disc_real_b, disc_fake_b)

        # Get the gradients for the discriminators and update the weights
        self.disc_a_optimizer.apply(tape.gradient(total_disc_a_loss, self.disc_a.trainable_variables), self.disc_a.trainable_variables)
        self.disc_b_optimizer.apply(tape.gradient(total_disc_b_loss, self.disc_b.trainable_variables), self.disc_b.trainable_variables)

        # Update metrics and return their value.
        self.d_a_loss_tracker.update_state(total_disc_a_loss)
        self.d_b_loss_tracker.update_state(total_disc_b_loss)
        self.d_fake_a_loss_tracker.update_state(disc_a_loss_fake)
        self.d_fake_b_loss_tracker.update_state(disc_b_loss_fake)
        self.d_real_a_loss_tracker.update_state(disc_a_loss_real)
        self.d_real_b_loss_tracker.update_state(disc_b_loss_real)
        self.g_a_loss_tracker.update_state(total_gen_loss_a)
        self.g_b_loss_tracker.update_state(total_gen_loss_b)
        self.g_adv_a_loss_tracker.update_state(adv_loss_a)
        self.g_adv_b_loss_tracker.update_state(adv_loss_b)
        self.g_cyc_a_loss_tracker.update_state(cycle_loss_a)
        self.g_cyc_b_loss_tracker.update_state(cycle_loss_b)
        self.g_id_a_loss_tracker.update_state(id_loss_a)
        self.g_id_b_loss_tracker.update_state(id_loss_b)

        return {m.name: m.result() for m in self.metrics}

    @staticmethod
    def to_numpy_array(x):
        """Converts torch tensor to numpy."""
        if os.environ["KERAS_BACKEND"] == "torch":
            if torch.cuda.is_available():
                x = x.cpu()
            return x.data.numpy().copy()
        elif os.environ["KERAS_BACKEND"] == "tensorflow":
            return x.numpy().copy()


class GANMonitor(keras.callbacks.Callback):
    """A callback to generate and save images after each epoch"""
    def __init__(self, test_a, test_b, output_dir, num_img=2):
        super(GANMonitor, self).__init__()
        self.num_img = num_img
        self.test_a = test_a
        self.test_b = test_b
        self.output_dir = output_dir

    def on_epoch_end(self, epoch, logs=None):
        self.plot_reconstruction(self.model, epoch+1, nex=self.num_img)

    def plot_reconstruction(self, model, epoch, nex=2):
        w = self.test_a.shape[2]
        h = self.test_a.shape[1]
        img_aba = np.zeros((min(nex, len(self.test_a)) * h, 4 * w, 3), dtype='uint8')
        img_bab = np.zeros_like(img_aba)
        brightness_factor_overlay = 0.7
        
        for i in range(0, min(nex, len(self.test_a))):
            img = self.test_a[i, :, :, 0]
            img -= np.min(img)
            img /= np.max(img)
            img *= 255
            img = img.astype(np.uint8)
            for j in range(0, 3):
                img_aba[i*h: (i+1)*h, 0: w, j] = img[:, :]

            prediction = model.gen_a(self.test_a[i:i + 1, :, :, :])
            img = CycleGanModel.to_numpy_array(prediction)[0]
            img = img[:, :, 0]
            img -= np.min(img)
            img /= np.max(img)
            img *= 255
            img = img.astype(np.uint8)
            msk = (img.copy() > 127)
            for j in range(0, 3):
                img_aba[i*h: (i+1)*h, w: 2*w, j] = img[:, :]

            prediction = model.gen_b(prediction)
            img = CycleGanModel.to_numpy_array(prediction)[0]
            img = img[:, :, 0]
            img -= np.min(img)
            img /= np.max(img)
            img *= 255
            img = img.astype(np.uint8)
            for j in range(0, 3):
                img_aba[i*h: (i+1)*h, 2*w: 3*w, j] = img[:, :]
                
            msk ^= ndimage.binary_erosion(msk, iterations=2)
            for j in range(0, 3):
                if j == 0:
                    img_aba[i*h: (i+1)*h, 3*w: 4*w, j] = np.maximum((img_aba[i*h: (i+1)*h, 0: w, 0]*brightness_factor_overlay).astype('uint8'), msk*255)
                else:
                    img_aba[i*h: (i+1)*h, 3*w: 4*w, j] = (img_aba[i*h: (i+1)*h, 0: w, 0]*brightness_factor_overlay).astype('uint8')

        Image.fromarray(img_aba).save(os.path.join(self.output_dir, 'A-B-A_Epoch_{:05d}.tif'.format(epoch)))
        
        for i in range(0, min(nex, len(self.test_a))):
            img = self.test_b[i, :, :, 0]
            img -= np.min(img)
            img /= np.max(img)
            img *= 255
            img = img.astype(np.uint8)
            msk = (img.copy() > 127)
            for j in range(0, 3):
                img_bab[i*h: (i+1)*h, 0: w, j] = img[:, :]

            prediction = model.gen_b(self.test_b[i:i + 1, :, :, :])
            img = CycleGanModel.to_numpy_array(prediction)[0]
            img = img[:, :, 0]
            img -= np.min(img)
            img /= np.max(img)
            img *= 255
            img = img.astype(np.uint8)
            for j in range(0, 3):
                img_bab[i*h: (i+1)*h, w: 2*w, j] = img[:, :]

            prediction = model.gen_a(prediction)
            img = CycleGanModel.to_numpy_array(prediction)[0]
            img = img[:, :, 0]
            img -= np.min(img)
            img /= np.max(img)
            img *= 255
            img = img.astype(np.uint8)
            for j in range(0, 3):
                img_bab[i*h: (i+1)*h, 2*w: 3*w, j] = img[:, :]
            
            msk ^= ndimage.binary_erosion(msk, iterations=2)
            for j in range(0, 3):
                if j == 0:
                    img_bab[i*h: (i+1)*h, 3*w: 4*w, j] = np.maximum((img_bab[i*h: (i+1)*h, w: 2*w, 0]*brightness_factor_overlay).astype('uint8'), msk*255)
                else:
                    img_bab[i*h: (i+1)*h, 3*w: 4*w, j] = (img_bab[i*h: (i+1)*h, w: 2*w, 0]*brightness_factor_overlay).astype('uint8')

        Image.fromarray(img_bab).save(os.path.join(self.output_dir, 'B-A-B_Epoch_{:05d}.tif'.format(epoch)))


class ImagePool:
    """This class implements an image buffer that stores previously generated images.

    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators.
    """

    def __init__(self, batch_size, pool_size=50):
        """Initialize the ImagePool class

        Parameters:
            pool_size (int) -- the size of image buffer, if pool_size=0, no buffer will be created
        """
        self.pool_size = pool_size
        self.batch_size = batch_size
        if self.pool_size > 0:  # create an empty pool
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        """Return an image from the pool.

        Parameters:
            images: the latest generated images from the generator

        Returns images from the buffer.

        By 50/100, the buffer will return input images.
        By 50/100, the buffer will return images previously stored in the buffer,
        and insert the current images to the buffer.
        """
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return images

        return_images = []

        for index in range(0, self.batch_size):
            try:
                image = keras.ops.expand_dims(images[index], 0)
            except ValueError:
                break  # Out of bounds error in images[index]. Can occur on the last batch if there are less than batch_size elements in the tensor. In that case, just stop the loop.

            if self.num_imgs < self.pool_size:   # if the buffer is not full; keep inserting current images to the buffer
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = keras.ops.copy(self.images[random_id])
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:       # by another 50% chance, the buffer will return the current image
                    return_images.append(image)
        return_images = keras.ops.concatenate(return_images, 0)
        return return_images


if __name__ == '__main__':
    cycle_gan = CycleGAN(root_dir='./')
