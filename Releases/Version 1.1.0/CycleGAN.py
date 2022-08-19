"""
Based on https://github.com/keras-team/keras-io/blob/master/examples/generative/cyclegan.py
"""
import os
import time
from PIL import Image
from scipy import ndimage
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tqdm import tqdm

import HelperFunctions


class CycleGAN:
    def __init__(self, root_dir='./', image_shape=(384, 384, 1), allow_memory_growth=True, use_gpus_no=(0, )):
        # Training and hyperparameters
        self.batch_size = 2
        self.epochs = 40
        self.learning_rate = 2e-4
        self.use_data_loader = False
        self.filters = 32
        self.num_downsampling_blocks_gen = 2
        self.num_residual_blocks_gen = 9
        self.num_upsampling_blocks_gen = 2
        self.num_downsampling_blocks_disc = 2

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

        # Losses and Loss Weights; cycle and identity loss for A->B (images->binary masks) can be set to BinaryCrossentropy instead of MeanAbsoluteError, but this makes the network asymmetric (last layer in Generator_A (i.e., the generator that generates B from A or Masks from Images) will be sigmoid instead of tanh), and it might be necessary to adjust lambda_cylce_A (and lambda_identity_A if used)
        self.lambda_cylce_a = 10
        self.lambda_cylce_b = 10
        self.use_binary_crossentropy = False

        # Linear decay of learning rate, for both discriminators and generators
        self.use_linear_decay = True
        self.decay_epoch = int(0.75 * self.epochs)  # The epoch where the linear decay of the learning rates start

        # Identity loss - send images from B to G_A2B (and the opposite) to teach identity mappings; set to value > 0 (e.g., 0.5) to enable identity mapping - not compatible with use_binary_crossentropy
        self.lambda_identity_a = 0.0
        self.lambda_identity_b = 0.0
        assert not (self.use_binary_crossentropy and (self.lambda_identity_a > 0 or self.lambda_identity_a > 0)), 'In the current implementation, binary crossentropy cannot be used with identity mapping. Please set either self.use_binary_crossentropy = False or both iudentity losses to 0.'

        # Skip Connection - adds a skip connection between the input and output in the generator (conceptually similar to an identity mapping)
        self.use_skip_connection = False

        # Resize convolution - instead of transpose convolution in deconvolution layers - can reduce checkerboard artifacts but the blurring might affect the cycle-consistency
        self.use_resize_convolution = False
        
        # Discriminator regularization - avoid "overtraining" the discriminator
        self.label_smoothing_factor = 0.0  # Label smoothing factor - set to a small value (e.g., 0.1) to avoid overconfident discriminator guesses and very low discriminator losses (too strong discriminators can be problematic for generators due to adverserial nature of GANs)
        self.gaussian_noise_value = 0.15   # Set to a small value (e.g., 0.15) to add Gaussian Noise to the discriminator layers (can help against mode collapse and "overtraining" the discriminator)

        # Set up directories and variables
        self.gen_a = None
        self.gen_b = None
        self.disc_x = None
        self.disc_y = None
        self.adv_loss_fn = None
        self.model = None
        self.data = None
        self.kernel_init = None
        self.gamma_init = None
        self.root_dir = root_dir
        self.model_dir = os.path.join(self.root_dir, '2_CycleGAN', 'Models')
        self.image_shape = image_shape
        self.prefix = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
        self.cycle_loss_fn_a = tf.keras.losses.MeanAbsoluteError()
        self.cycle_loss_fn_b = tf.keras.losses.MeanAbsoluteError()
        self.identity_loss_fn_a = tf.keras.losses.MeanAbsoluteError()
        self.identity_loss_fn_b = tf.keras.losses.MeanAbsoluteError()

        # Arrays for training and test data
        self.train_a = HelperFunctions.get_image_file_paths_from_directory(os.path.join(self.root_dir, '2_CycleGAN', 'data', 'trainA'))
        self.test_a = HelperFunctions.get_image_file_paths_from_directory(os.path.join(self.root_dir, '2_CycleGAN', 'data', 'testA'))
        self.train_b = HelperFunctions.get_image_file_paths_from_directory(os.path.join(self.root_dir, '2_CycleGAN', 'data', 'trainB'))
        self.test_b = HelperFunctions.get_image_file_paths_from_directory(os.path.join(self.root_dir, '2_CycleGAN', 'data', 'testB'))

    def create_model(self):
        if self.use_binary_crossentropy:
            self.cycle_loss_fn_a = tf.keras.losses.BinaryCrossentropy()
            self.cycle_loss_fn_b = tf.keras.losses.MeanAbsoluteError()
            self.identity_loss_fn_a = tf.keras.losses.BinaryCrossentropy()
            self.identity_loss_fn_b = tf.keras.losses.MeanAbsoluteError()

        # Weights initializer for the layers.
        # tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
        self.kernel_init = tf.keras.initializers.GlorotUniform()
        # Gamma initializer for instance normalization.
        # self.gamma_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
        self.gamma_init = 'ones'
        # Loss function for evaluating adversarial loss
        self.adv_loss_fn = tf.keras.losses.MeanSquaredError()

        # Build the generators and discriminators
        self.gen_a = self.get_resnet_generator(name="generator_A",
                                               filters=self.filters,
                                               num_downsampling_blocks=self.num_downsampling_blocks_gen,
                                               num_residual_blocks=self.num_residual_blocks_gen,
                                               num_upsample_blocks=self.num_upsampling_blocks_gen,
                                               use_binary_crossentropy=self.use_binary_crossentropy)
        self.gen_b = self.get_resnet_generator(name="generator_B",
                                               filters=self.filters,
                                               num_downsampling_blocks=self.num_downsampling_blocks_gen,
                                               num_residual_blocks=self.num_residual_blocks_gen,
                                               num_upsample_blocks=self.num_upsampling_blocks_gen,
                                               use_binary_crossentropy=False)

        self.disc_x = self.get_discriminator(name="discriminator_X", num_downsampling_blocks=self.num_downsampling_blocks_disc, filters=2 * self.filters)
        self.disc_y = self.get_discriminator(name="discriminator_Y", num_downsampling_blocks=self.num_downsampling_blocks_disc, filters=2 * self.filters)

        # Create CycleGAN model
        model = CycleGanModel(
            generator_a=self.gen_a,
            generator_b=self.gen_b,
            discriminator_x=self.disc_x,
            discriminator_y=self.disc_y,
            lambda_cycle_a=self.lambda_cylce_a,
            lambda_cycle_b=self.lambda_cylce_b,
            lambda_identity_a=self.lambda_identity_a,
            lambda_identity_b=self.lambda_identity_b,
        )

        # Compile the model
        model.compile(
            gen_a_optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=0.5),
            gen_b_optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=0.5),
            disc_x_optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=0.5),
            disc_y_optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=0.5),
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

        self.test_a = self.load_images(self.test_a, scale_for_binary_crossentropy=False)
        self.test_b = self.load_images(self.test_b, scale_for_binary_crossentropy=self.use_binary_crossentropy)
        if not self.use_data_loader:
            self.train_a = self.load_images(self.train_a, scale_for_binary_crossentropy=False)
            self.train_b = self.load_images(self.train_b, scale_for_binary_crossentropy=self.use_binary_crossentropy)

        self.data = DataLoader(self.train_a, self.train_b, batch_size=self.batch_size, use_dataloader=self.use_data_loader, scale_for_binary_crossentropy=self.use_binary_crossentropy)

        # Create the model
        self.model = self.create_model()

        # Callbacks
        plotter = GANMonitor(self.test_a, self.test_b, output_dir=os.path.join(self.root_dir, '2_CycleGAN', 'images', self.prefix), num_img=2)
        checkpoint_filepath = os.path.join(self.model_dir, self.prefix, 'checkpoints_{epoch:03d}')
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath)
        csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(self.model_dir, self.prefix, 'training_log.csv'), separator=';', append=True)
        callbacks_list = [plotter, model_checkpoint_callback, csv_logger]
        if self.use_linear_decay:
            rate_scheduler = tf.keras.callbacks.LearningRateScheduler(self.linear_decay)
            callbacks_list.append(rate_scheduler)

        # Train the model
        self.model.fit(
            self.data,
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=callbacks_list,
        )

        self.model.save(os.path.join(self.model_dir,  self.prefix))
        return self.model

    def run_inference(self, files, output_directory, source_domain, model=None, tile_images=False, min_overlap=2, manage_overlap_mode=2, use_gpu=False):
        if self.model is None:
            if model is None:
                # Load the most recent model
                self.model = tf.keras.models.load_model(os.path.join(self.model_dir, os.listdir(self.model_dir)[-1]))
            elif isinstance(self.model, str):
                # Load the specified model
                self.model = tf.keras.models.load_model(model)
            else:
                self.model = model

        if use_gpu:
            device = tf.config.list_logical_devices('GPU')[0]
        else:
            device = tf.config.list_logical_devices('CPU')[0]

        if 'a' in source_domain.lower():
            generator_model = self.model.gen_a
        else:
            generator_model = self.model.gen_b

        input_files = HelperFunctions.load_and_preprocess_images(files, normalization_range=(-1, 1))
        file_names = HelperFunctions.get_image_file_paths_from_directory(files)

        if not tile_images and input_files[0].shape != self.image_shape:
            self.image_shape = input_files[0].shape
            if 'a' in source_domain.lower():
                generator_model_new = self.get_resnet_generator(name="generator_a", filters=self.filters, use_binary_crossentropy=self.use_binary_crossentropy)
            else:
                generator_model_new = self.get_resnet_generator(name="generator_b", filters=self.filters, use_binary_crossentropy=False)
            generator_model_new.set_weights(generator_model.get_weights())
            generator_model = generator_model_new

        with tf.device(device.name):
            for i in tqdm(range(0, input_files.shape[0])):
                input_file = input_files[i]
                if tile_images:
                    tiles = np.asarray(HelperFunctions.tile_image(input_file, self.image_shape[0], self.image_shape[1], min_overlap=min_overlap))
                    prediction = np.asarray([generator_model(img.reshape(1, img.shape[0], img.shape[1], img.shape[2]))[0].numpy() for img in tiles])
                    img = HelperFunctions.stitch_image(prediction, input_file.shape[1], input_file.shape[0], min_overlap=min_overlap, manage_overlap_mode=manage_overlap_mode)
                else:
                    prediction = generator_model(input_file.reshape(1, input_file.shape[0], input_file.shape[1], input_file.shape[2]))
                    img = prediction[0].numpy().copy()
                img = img[:, :, 0]
                img -= np.min(img)
                img /= np.max(img)
                img *= 255
                img = img.astype(np.uint8)
                Image.fromarray(img).save(os.path.join(output_directory, os.path.split(file_names[i])[-1]))

    @staticmethod
    def load_images(image_list, scale_for_binary_crossentropy=False):
        if scale_for_binary_crossentropy:
            r = (0, 1)
        else:
            r = (-1, 1)
        images = HelperFunctions.load_and_preprocess_images(input_dir_or_filelist=image_list, threshold_value=None, normalization_range=r, output_channels=1, contrast_optimization_range=None)

        return images

    def generator_loss_fn(self, fake):
        fake_loss = self.adv_loss_fn(tf.ones_like(fake) * (1-self.label_smoothing_factor) + (self.label_smoothing_factor/2), fake)
        return fake_loss

    def discriminator_loss_fn(self, real, fake):
        real_loss = self.adv_loss_fn(tf.ones_like(real) * (1-self.label_smoothing_factor) + (self.label_smoothing_factor/2), real)
        fake_loss = self.adv_loss_fn(tf.zeros_like(fake) * (1-self.label_smoothing_factor) + (self.label_smoothing_factor/2), fake)
        return (real_loss + fake_loss) * 0.5

    def linear_decay(self, epoch):
        if epoch < self.decay_epoch:
            return self.learning_rate
        power = 1  # 1 -> Linear Decay
        return self.learning_rate * (1 - ((epoch-self.decay_epoch) / float(self.epochs-self.decay_epoch))) ** power

    ########################
    # Network Architecture #
    ########################

    def residual_block(self, input_tensor, activation, kernel_size=(3, 3), strides=(1, 1), padding="valid", use_bias=False):
        dim = input_tensor.shape[-1]

        x = ReflectionPadding2D()(input_tensor)
        x = tf.keras.layers.Conv2D(dim, kernel_size, strides=strides, kernel_initializer=self.kernel_init, padding=padding, use_bias=use_bias)(x)
        x = tfa.layers.InstanceNormalization(axis=3, center=True, gamma_initializer=self.gamma_init, epsilon=1e-5)(x, training=True)
        x = activation(x)

        x = ReflectionPadding2D()(x)
        x = tf.keras.layers.Conv2D(dim, kernel_size, strides=strides, kernel_initializer=self.kernel_init, padding=padding, use_bias=use_bias)(x)
        x = tfa.layers.InstanceNormalization(axis=3, center=True, gamma_initializer=self.gamma_init, epsilon=1e-5)(x, training=True)
        x = tf.keras.layers.add([input_tensor, x])
        return x

    def downsample(self, x, filters, activation, kernel_size=(3, 3), strides=(2, 2), padding="same", use_bias=False):
        x = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, kernel_initializer=self.kernel_init, padding=padding, use_bias=use_bias)(x)
        x = tfa.layers.InstanceNormalization(axis=3, center=True, gamma_initializer=self.gamma_init, epsilon=1e-5)(x, training=True)
        if activation:
            x = activation(x)
        return x

    def upsample(self, x, filters, activation, kernel_size=(3, 3), strides=(2, 2), padding="same", use_bias=False):
        if self.use_resize_convolution:
            x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)  # Nearest neighbor upsampling
            x = ReflectionPadding2D()(x)
            x = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=1, padding='valid', use_bias=use_bias)(x)
        else:
            x = tf.keras.layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding, kernel_initializer=self.kernel_init, use_bias=use_bias)(x)
        x = tfa.layers.InstanceNormalization(axis=3, center=True, gamma_initializer=self.gamma_init, epsilon=1e-5)(x, training=True)
        if activation:
            x = activation(x)
        return x

    def get_resnet_generator(self, filters=64, num_downsampling_blocks=2, num_residual_blocks=9, num_upsample_blocks=2, name=None, use_binary_crossentropy=False):
        # Input
        img_input = tf.keras.layers.Input(shape=self.image_shape, name=name + "_img_input")

        # Make sure the input has a size that can be downsampled at least num_downsampling_blocks times (if not, use reflection padding)
        padding_height = ((2 ** num_downsampling_blocks - img_input.shape[1] % 2 ** num_downsampling_blocks) % 2 ** num_downsampling_blocks)
        padding_width = ((2 ** num_downsampling_blocks - img_input.shape[2] % 2 ** num_downsampling_blocks) % 2 ** num_downsampling_blocks)
        x = ReflectionPadding2D((padding_width, padding_height))(img_input)

        x = ReflectionPadding2D(padding=(6, 6))(x)
        # 7 x 7 Convolution
        x = tf.keras.layers.Conv2D(filters, (7, 7), kernel_initializer=self.kernel_init, use_bias=False)(x)
        x = tfa.layers.InstanceNormalization(axis=3, center=True, gamma_initializer=self.gamma_init, epsilon=1e-5)(x, training=True)
        x = tf.keras.layers.Activation("relu")(x)

        # Downsampling
        for _ in range(num_downsampling_blocks):
            filters *= 2
            x = self.downsample(x, filters=filters, activation=tf.keras.layers.Activation("relu"))

        # Residual blocks
        for _ in range(num_residual_blocks):
            x = self.residual_block(x, activation=tf.keras.layers.Activation("relu"))

        # Upsampling
        for _ in range(num_upsample_blocks):
            filters //= 2
            x = self.upsample(x, filters, activation=tf.keras.layers.Activation("relu"))

        # Final block
        x = ReflectionPadding2D(padding=(6, 6))(x)
        x = tf.keras.layers.Conv2D(self.image_shape[-1], (7, 7), padding="valid")(x)
        x = tf.keras.layers.Cropping2D(cropping=((padding_height // 2, padding_height // 2 + padding_height % 2), (padding_width // 2, padding_width // 2 + padding_width % 2)))(x)

        if self.use_skip_connection:
            shortcut = img_input
            shortcut = tf.keras.layers.Conv2D(filters, (1, 1), kernel_initializer=self.kernel_init, activation=None, use_bias=False)(shortcut)
            shortcut = tfa.layers.InstanceNormalization(axis=3, center=True, gamma_initializer=self.gamma_init, epsilon=1e-5)(shortcut, training=True)
            shortcut = tf.keras.layers.Activation("relu")(shortcut)

            out = ReflectionPadding2D()(img_input)
            out = tf.keras.layers.Conv2D(filters, (3, 3), kernel_initializer=self.kernel_init, use_bias=False)(out)
            out = tfa.layers.InstanceNormalization(axis=3, center=True, gamma_initializer=self.gamma_init, epsilon=1e-5)(out, training=True)
            out = tf.keras.layers.Activation("relu")(out)

            out = tf.keras.layers.add([shortcut, out])
            out = tfa.layers.InstanceNormalization(axis=3, center=True, gamma_initializer=self.gamma_init, epsilon=1e-5)(out, training=True)
            out = tf.keras.layers.Activation("relu")(out)

            x = tf.keras.layers.concatenate([out, x], axis=3)
            x = tf.keras.layers.Conv2D(self.image_shape[-1], (1, 1), kernel_initializer=self.kernel_init, activation=None, use_bias=False)(x)

        if use_binary_crossentropy:
            x = tf.keras.layers.Activation("sigmoid", name=f'output_{name}')(x)
        else:
            x = tf.keras.layers.Activation("tanh", name=f'output_{name}')(x)

        model = tf.keras.models.Model(img_input, x, name=name)
        return model

    def get_discriminator(self, filters=32, num_downsampling_blocks=3, name=None):
        img_input = tf.keras.layers.Input(shape=self.image_shape, name=name + "_img_input")
        if self.gaussian_noise_value > 0:
            x = tf.keras.layers.GaussianNoise(self.gaussian_noise_value)(img_input)
            x = tf.keras.layers.Conv2D(filters, (4, 4), strides=(2, 2), padding="same", kernel_initializer=self.kernel_init)(x)
        else:
            x = tf.keras.layers.Conv2D(filters, (4, 4), strides=(2, 2), padding="same", kernel_initializer=self.kernel_init)(img_input)

        x = tf.keras.layers.LeakyReLU(0.2)(x)

        for num_downsample_block in range(num_downsampling_blocks):
            filters *= 2
            if num_downsample_block < 2:
                if self.gaussian_noise_value > 0:
                    x = tf.keras.layers.GaussianNoise(self.gaussian_noise_value)(x)
                x = self.downsample(x, filters=filters, activation=tf.keras.layers.LeakyReLU(0.2), kernel_size=(4, 4), strides=(2, 2))
            else:
                if self.gaussian_noise_value > 0:
                    x = tf.keras.layers.GaussianNoise(self.gaussian_noise_value)(x)
                x = self.downsample(x, filters=filters, activation=tf.keras.layers.LeakyReLU(0.2), kernel_size=(4, 4), strides=(1, 1))

        if self.gaussian_noise_value > 0:
            x = tf.keras.layers.GaussianNoise(self.gaussian_noise_value)(x)
        x = tf.keras.layers.Conv2D(1, (4, 4), strides=(1, 1), padding="same", kernel_initializer=self.kernel_init, name=f'output_{name}')(x)

        model = tf.keras.models.Model(inputs=img_input, outputs=x, name=name)
        return model


class DataLoader(tf.keras.utils.Sequence):
    def __init__(self, train_a, train_b, batch_size=1, use_dataloader=False, scale_for_binary_crossentropy=False):
        self.batch_size = batch_size
        self.train_a = train_a
        self.train_b = train_b
        self.use_dataloader = use_dataloader
        self.scale_for_binary_crossentropy = scale_for_binary_crossentropy

    def __len__(self):
        return int(max(len(self.train_a), len(self.train_b)) / float(self.batch_size))

    def __getitem__(self, idx):
        real_images_a = self.train_a[idx * self.batch_size:(idx + 1) * self.batch_size]
        real_images_b = self.train_b[idx * self.batch_size:(idx + 1) * self.batch_size]

        if self.use_dataloader:
            real_images_a = CycleGAN.load_images(real_images_a, False)
            real_images_b = CycleGAN.load_images(real_images_b, self.scale_for_binary_crossentropy)

        return np.asarray(real_images_a), np.asarray(real_images_b)
        
    def on_epoch_end(self):
        np.random.shuffle(self.train_a)
        np.random.shuffle(self.train_b)


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


class CycleGanModel(tf.keras.Model):
    def __init__(self, generator_a, generator_b, discriminator_x, discriminator_y, lambda_cycle_a=10.0, lambda_cycle_b=10.0, lambda_identity_a=0.5, lambda_identity_b=0.5):
        super(CycleGanModel, self).__init__()
        self.gen_a = generator_a
        self.gen_b = generator_b
        self.disc_x = discriminator_x
        self.disc_y = discriminator_y
        self.lambda_cycle_a = lambda_cycle_a
        self.lambda_cycle_b = lambda_cycle_b
        self.lambda_identity_a = lambda_identity_a
        self.lambda_identity_b = lambda_identity_b

        self.use_identity_loss = lambda_identity_a > 0 or lambda_identity_b > 0

        self.gen_a_optimizer = None
        self.gen_b_optimizer = None
        self.disc_x_optimizer = None
        self.disc_y_optimizer = None
        self.generator_loss_fn = None
        self.discriminator_loss_fn = None
        self.cycle_loss_fn_a = None
        self.cycle_loss_fn_b = None
        self.identity_loss_fn_a = None
        self.identity_loss_fn_b = None

    def call(self, inputs):
        return self.disc_x(inputs), self.disc_y(inputs), self.gen_a(inputs), self.gen_b(inputs)

    def compile(self, gen_a_optimizer, gen_b_optimizer, disc_x_optimizer, disc_y_optimizer, disc_loss_fn, gen_loss_fn, cycle_loss_fn_a=tf.keras.losses.MeanAbsoluteError(), cycle_loss_fn_b=tf.keras.losses.MeanAbsoluteError(), identity_loss_fn_a=tf.keras.losses.MeanAbsoluteError(), identity_loss_fn_b=tf.keras.losses.MeanAbsoluteError()):
        super(CycleGanModel, self).compile()
        self.gen_a_optimizer = gen_a_optimizer
        self.gen_b_optimizer = gen_b_optimizer
        self.disc_x_optimizer = disc_x_optimizer
        self.disc_y_optimizer = disc_y_optimizer
        self.generator_loss_fn = gen_loss_fn
        self.discriminator_loss_fn = disc_loss_fn
        self.cycle_loss_fn_a = cycle_loss_fn_a
        self.cycle_loss_fn_b = cycle_loss_fn_b
        self.identity_loss_fn_a = identity_loss_fn_a
        self.identity_loss_fn_b = identity_loss_fn_b

    def train_step(self, batch_data):
        # x is Domain1 and y is Domain2
        real_x, real_y = batch_data

        # For CycleGAN, we need to calculate different
        # kinds of losses for the generators and discriminators.
        # We will perform the following steps here:
        #
        # 1. Pass real images through the generators and get the generated images
        # 2. Pass the generated images back to the generators to check if
        #    we can predict the original image from the generated image.
        # 3. Do an identity mapping of the real images using the generators.
        # 4. Pass the generated images in 1) to the corresponding discriminators.
        # 5. Calculate the generators total loss (adverserial + cycle + identity)
        # 6. Calculate the discriminators loss
        # 7. Update the weights of the generators
        # 8. Update the weights of the discriminators
        # 9. Return the losses in a dictionary

        with tf.GradientTape(persistent=True) as tape:
            # X to fake Y
            fake_y = self.gen_a(real_x, training=True)
            # Y to fake X -> y2x
            fake_x = self.gen_b(real_y, training=True)

            # Cycle (X to fake Y to fake X): x -> y -> x
            cycled_x = self.gen_b(fake_y, training=True)
            # Cycle (Y to fake X to fake Y) y -> x -> y
            cycled_y = self.gen_a(fake_x, training=True)

            # Identity mapping
            if self.use_identity_loss:
                same_x = self.gen_b(real_x, training=True)
                same_y = self.gen_a(real_y, training=True)

            # Discriminator output
            disc_real_x = self.disc_x(real_x, training=True)
            disc_fake_x = self.disc_x(fake_x, training=True)

            disc_real_y = self.disc_y(real_y, training=True)
            disc_fake_y = self.disc_y(fake_y, training=True)

            # Generator adverserial loss
            gen_a_loss = self.generator_loss_fn(disc_fake_y)
            gen_b_loss = self.generator_loss_fn(disc_fake_x)

            # Generator cycle loss
            cycle_loss_a = self.cycle_loss_fn_a(real_y, cycled_y) * self.lambda_cycle_a
            cycle_loss_b = self.cycle_loss_fn_b(real_x, cycled_x) * self.lambda_cycle_b

            # Generator identity loss
            if self.use_identity_loss:
                id_loss_a = (
                        self.identity_loss_fn_a(real_y, same_y)
                        * self.lambda_cycle_a
                        * self.lambda_identity_a
                )
                id_loss_b = (
                        self.identity_loss_fn_b(real_x, same_x)
                        * self.lambda_cycle_b
                        * self.lambda_identity_b
                )
            else:
                id_loss_a = 0
                id_loss_b = 0

            # Total generator loss
            total_loss_a = gen_a_loss + cycle_loss_a + id_loss_a
            total_loss_b = gen_b_loss + cycle_loss_b + id_loss_b

            # Discriminator loss
            disc_x_loss = self.discriminator_loss_fn(disc_real_x, disc_fake_x)
            disc_y_loss = self.discriminator_loss_fn(disc_real_y, disc_fake_y)

        # Get the gradients for the generators
        grads_a = tape.gradient(total_loss_a, self.gen_a.trainable_variables)
        grads_b = tape.gradient(total_loss_b, self.gen_b.trainable_variables)

        # Get the gradients for the discriminators
        disc_x_grads = tape.gradient(disc_x_loss, self.disc_x.trainable_variables)
        disc_y_grads = tape.gradient(disc_y_loss, self.disc_y.trainable_variables)

        # Update the weights of the generators
        self.gen_a_optimizer.apply_gradients(
            zip(grads_a, self.gen_a.trainable_variables)
        )
        self.gen_b_optimizer.apply_gradients(
            zip(grads_b, self.gen_b.trainable_variables)
        )

        # Update the weights of the discriminators
        self.disc_x_optimizer.apply_gradients(
            zip(disc_x_grads, self.disc_x.trainable_variables)
        )
        self.disc_y_optimizer.apply_gradients(
            zip(disc_y_grads, self.disc_y.trainable_variables)
        )

        return {"G_A_loss": total_loss_a, "G_B_loss": total_loss_b, "D_X_loss": disc_x_loss, "D_Y_loss": disc_y_loss}
    

class GANMonitor(tf.keras.callbacks.Callback):
    """A callback to generate and save images after each epoch"""

    def __init__(self, test_a, test_b, output_dir, num_img=2):
        self.num_img = num_img
        self.test_a = test_a
        self.test_b = test_b
        self.output_dir = output_dir

    def on_epoch_end(self, epoch, logs=None):
        self.plot_reconstruction(self.model, epoch, nex=self.num_img)

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
            img = prediction[0].numpy().copy()
            img = img[:, :, 0]
            img -= np.min(img)
            img /= np.max(img)
            img *= 255
            img = img.astype(np.uint8)
            msk = (img.copy() > 127)
            for j in range(0, 3):
                img_aba[i*h: (i+1)*h, w: 2*w, j] = img[:, :]

            prediction = model.gen_b(prediction)
            img = prediction[0].numpy().copy()
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
            img = prediction[0].numpy().copy()
            img = img[:, :, 0]
            img -= np.min(img)
            img /= np.max(img)
            img *= 255
            img = img.astype(np.uint8)
            for j in range(0, 3):
                img_bab[i*h: (i+1)*h, w: 2*w, j] = img[:, :]

            prediction = model.gen_a(prediction)
            img = prediction[0].numpy().copy()
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


if __name__ == '__main__':
    cycle_gan = CycleGAN(root_dir='./')
