import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm.autonotebook import tqdm
import opensimplex

import os
from shutil import copy
from PIL import Image
import random
import cv2
import math
from scipy import ndimage
import keras

import HelperFunctions

if os.environ["KERAS_BACKEND"] == "torch":
    import torch
elif os.environ["KERAS_BACKEND"] == "tensorflow":
    import tensorflow as tf
else:
    raise NotImplementedError('Unsupported backend. Please make sure the environment variable "KERAS_BACKEND" is set either to "tensorflow" or to "torch".')


@keras.saving.register_keras_serializable()
class WGAN_GP(keras.Model):
    def __init__(
            self,
            discriminator,
            generator,
            latent_dim,
            discriminator_extra_steps=3,
            gp_weight=10.0,
            **kwargs
    ):
        super(WGAN_GP, self).__init__(**kwargs)
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.d_steps = discriminator_extra_steps
        self.gp_weight = gp_weight
        self.d_optimizer = None
        self.g_optimizer = None
        self.d_loss_fn = None
        self.g_loss_fn = None

        self.losses_log = {'d_loss': [], 'g_loss': [], 'grad_penalty': [], 'grad_norm': []}
        self.d_loss_tracker = keras.metrics.Mean(name="d_loss")
        self.d_total_loss_tracker = keras.metrics.Mean(name="d_total_loss")
        self.g_loss_tracker = keras.metrics.Mean(name="g_loss")
        self.gp_loss_tracker = keras.metrics.Mean(name="grad_penalty")
        self.gn_loss_tracker = keras.metrics.Mean(name="grad_norm")

        self.built = True

    @property
    def metrics(self):
        return [self.d_loss_tracker, self.d_total_loss_tracker, self.g_loss_tracker, self.gp_loss_tracker, self.gn_loss_tracker]

    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn, **kwargs):
        super(WGAN_GP, self).compile(**kwargs)
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

    def call(self, inputs):
        return self.generator(inputs)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                'discriminator': self.discriminator,
                'generator': self.generator,
                'latent_dim': self.latent_dim,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        discriminator = keras.saving.deserialize_keras_object(config.pop('discriminator'))
        generator = keras.saving.deserialize_keras_object(config.pop('generator'))
        return cls(discriminator=discriminator, generator=generator, **config)

    def gradient_penalty(self, real_images, fake_images):
        """ Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # get the batch size
        batch_size = keras.ops.shape(real_images)[0]
        # get the interpolated image
        alpha = keras.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        if os.environ["KERAS_BACKEND"] == 'tensorflow':
            with tf.GradientTape() as gp_tape:
                gp_tape.watch(interpolated)
                # 1. Get the discriminator output for this interpolated image.
                pred = self.discriminator(interpolated, training=True)

            # 2. Calculate the gradients w.r.t to this interpolated image.
            grads = gp_tape.gradient(pred, [interpolated])[0]
        elif os.environ["KERAS_BACKEND"] == 'torch':
            # 1. Get the discriminator output for this interpolated image.
            pred = self.discriminator(interpolated, training=True)

            # 2. Calculate the gradients w.r.t to this interpolated image.
            grads = torch.autograd.grad(outputs=pred, inputs=interpolated, grad_outputs=torch.ones_like(pred), create_graph=True, retain_graph=True)[0]
            # # Gradients have shape (batch_size, num_channels, img_width, img_height), so flatten to easily take norm per example in batch
            # grads = grads.view(batch_size, -1)

        # 3. Calculate the norm of the gradients
        norm = keras.ops.sqrt(keras.ops.sum(keras.ops.square(grads), axis=[1, 2, 3]))
        gp = keras.ops.mean((norm - 1.0) ** 2)
        return gp, norm

    def train_step(self, real_images):
        if os.environ["KERAS_BACKEND"] == "torch":
            return self.train_step_torch(real_images)
        elif os.environ["KERAS_BACKEND"] == "tensorflow":
            return self.train_step_tensorflow(real_images)
        else:
            raise NotImplementedError('Unsupported backend. Please make sure the environment variable "KERAS_BACKEND" is set either to "tensorflow" or to "torch".')

    def train_step_tensorflow(self, real_images):
        if isinstance(real_images, tuple):
            real_images = real_images[0]
        batch_size = keras.ops.shape(real_images)[0]

        # Train discriminator first. The original paper recommends training the discriminator for `x` more steps (typically 5) as compared to one step of the generator.
        for i in range(self.d_steps):
            # Get the latent vector
            random_latent_vectors = keras.random.normal(shape=(batch_size, self.latent_dim))
            with tf.GradientTape() as tape:
                # Generate fake images from the latent vector
                fake_images = self.generator(random_latent_vectors, training=True)
                # Get the logits for the fake images
                fake_logits = self.discriminator(fake_images, training=True)
                # Get the logits for real images
                real_logits = self.discriminator(real_images, training=True)

                # Calculate discriminator loss using fake and real logits
                d_cost = self.d_loss_fn(real_img=real_logits, fake_img=fake_logits)
                # Calculate the gradient penalty
                gp, gn = self.gradient_penalty(real_images, fake_images)
                # Add the gradient penalty to the original discriminator loss
                d_loss = d_cost + gp * self.gp_weight

            # Get the gradients w.r.t the discriminator loss and update the weights
            self.d_optimizer.apply(tape.gradient(d_loss, self.discriminator.trainable_variables), self.discriminator.trainable_variables)

        # Train the generator now.
        # Get the latent vector
        random_latent_vectors = keras.random.normal(shape=(batch_size, self.latent_dim))
        with tf.GradientTape() as tape:
            # Generate fake images using the generator
            generated_images = self.generator(random_latent_vectors, training=True)
            # Get the discriminator logits for fake images
            gen_img_logits = self.discriminator(generated_images, training=True)
            # Calculate the generator loss
            g_loss = self.g_loss_fn(gen_img_logits)

        # Get the gradients w.r.t the generator loss and update the weights
        self.g_optimizer.apply(tape.gradient(g_loss, self.generator.trainable_variables), self.generator.trainable_variables)

        # Update metrics and return their value.
        self.g_loss_tracker.update_state(g_loss)
        self.d_loss_tracker.update_state(d_cost)
        self.gp_loss_tracker.update_state(gp)
        self.gn_loss_tracker.update_state(gn)
        self.d_total_loss_tracker.update_state(d_loss)

        return {m.name: m.result() for m in self.metrics}

    def train_step_torch(self, real_images):
        if isinstance(real_images, tuple):
            real_images = real_images[0]
        batch_size = keras.ops.shape(real_images)[0]

        # Train discriminator first. The original paper recommends training the discriminator for `x` more steps (typically 5) as compared to one step of the generator.
        for i in range(self.d_steps):
            # Get the latent vector
            random_latent_vectors = keras.random.normal(shape=(batch_size, self.latent_dim))

            # Generate fake images from the latent vector
            fake_images = self.generator(random_latent_vectors, training=True)
            # Get the logits for the fake images
            fake_logits = self.discriminator(fake_images, training=True)
            # Get the logits for real images
            real_logits = self.discriminator(real_images, training=True)

            # Calculate discriminator loss using fake and real logits
            d_cost = self.d_loss_fn(real_img=real_logits, fake_img=fake_logits)
            # Calculate the gradient penalty
            gp, gn = self.gradient_penalty(real_images, fake_images)
            # Add the gradient penalty to the original discriminator loss
            d_loss = d_cost + gp * self.gp_weight

            # Zero the gradients
            self.discriminator.zero_grad()

            # Get the gradients w.r.t the discriminator loss and update the weights
            d_loss.backward()
            with torch.no_grad():
                self.d_optimizer.apply([v.value.grad for v in self.discriminator.trainable_weights], self.discriminator.trainable_weights)

        # Train the generator now.
        # Get the latent vector
        random_latent_vectors = keras.random.normal(shape=(batch_size, self.latent_dim))
        # Generate fake images using the generator
        generated_images = self.generator(random_latent_vectors, training=True)
        # Get the discriminator logits for fake images
        gen_img_logits = self.discriminator(generated_images, training=True)
        # Calculate the generator loss
        g_loss = self.g_loss_fn(gen_img_logits)

        # Zero the gradients
        self.generator.zero_grad()

        # Get the gradients w.r.t the generator loss and update the weights
        g_loss.backward()
        with torch.no_grad():
            self.g_optimizer.apply([v.value.grad for v in self.generator.trainable_weights], self.generator.trainable_weights)

        # Update metrics and return their value.
        self.g_loss_tracker.update_state(g_loss)
        self.d_loss_tracker.update_state(d_cost)
        self.gp_loss_tracker.update_state(gp)
        self.gn_loss_tracker.update_state(gn)
        self.d_total_loss_tracker.update_state(d_loss)

        return {m.name: m.result() for m in self.metrics}

    @staticmethod
    def to_tensor(x):
        """Converts a numpy array to a torch tensor."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, device=device)
        return x

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
    def __init__(self, output_dir, num_img=9, latent_dim=128, output_epochs=100):
        super(GANMonitor, self).__init__()
        self.num_img = num_img
        self.latent_dim = latent_dim
        self.epochs = output_epochs
        self.output_dir = output_dir

    def on_epoch_end(self, epoch, logs=None):
        if epoch % int(self.epochs) == 0:
            self.plot_reconstruction(self.model, epoch, nex=self.num_img)

    def plot_reconstruction(self, model, epoch, nex=9):
        random_latent_vectors = keras.random.normal(shape=(nex, self.latent_dim))
        samples = WGAN_GP.to_numpy_array(model(random_latent_vectors))
        cols = 3
        rows = math.ceil(nex/float(cols))
        fig = plt.figure(figsize=(2, 2 * rows / cols))
        for i, s in enumerate(samples):
            fig.add_subplot(rows, cols, i + 1)
            plt.axis('off')
            plt.imshow((s*127.5+127.5)[:, :, 0].astype('uint8'), cmap="gray")

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'Epoch_{:05d}'.format(epoch)))
        plt.close(fig)


class WGAN:
    def __init__(self, root_dir, allow_memory_growth=True, use_gpus_no=(0, )):
        # Input and output directories
        self.root_dir = os.path.join(root_dir, '1_WGAN')
        self.input_dir = os.path.join(root_dir, 'Input_Masks')
        self.output_dir = os.path.join(self.root_dir, 'Output_Images')
        self.model_dir = os.path.join(self.root_dir, 'Models')
        self.generate_dir = os.path.join(root_dir, '2_CycleGAN', 'data', 'trainB')

        # Training parameters
        self.batch_size = 64
        self.epochs = 1000
        self.n_z = 128
        self.model = None

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

        # Create dataset
        self.train_images = []

        max_image_height = 0
        max_image_width = 0

        images = HelperFunctions.load_and_preprocess_images(input_dir_or_filelist=self.input_dir, threshold_value=0.5, normalization_range=(-1, 1), output_channels=1, contrast_optimization_range=None)
        for image in images:
            max_image_height = max([max_image_height, image.shape[0]])
            max_image_width = max([max_image_height, image.shape[1]])

            self.train_images.append(image.copy())
            self.train_images.append(np.fliplr(image.copy()))
            self.train_images.append(np.flipud(image.copy()))
            self.train_images.append(np.flipud(np.fliplr(image.copy())))        

        # Make sure the image has a size that can be downsampled at least 4 times (if not, pad with 0s)
        if max_image_height % 2**4 != 0:
            max_image_height = (max_image_height//(2**4) + 1) * 2**4
        if max_image_width % 2**4 != 0:
            max_image_width = (max_image_width//(2**4) + 1) * 2**4
        for i, image in enumerate(self.train_images):
            if image.shape[0] < max_image_height or image.shape[1] < max_image_width:
                img = np.zeros((max_image_height, max_image_width, 1), dtype='float32')
                img[(max_image_height-image.shape[0])//2:(max_image_height-image.shape[0])//2+image.shape[0], (max_image_width-image.shape[1])//2:(max_image_width-image.shape[1])//2+image.shape[1], :] = image[:, :, :]
                self.train_images[i] = img
        
        self.train_images = np.asarray(self.train_images, dtype='float32')

        self.prefix = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())

    def start_training(self):
        if not os.path.isdir(os.path.join(self.model_dir, self.prefix)):
            os.mkdir(os.path.join(self.model_dir, self.prefix))
        if not os.path.isdir(os.path.join(self.output_dir, self.prefix)):
            os.mkdir(os.path.join(self.output_dir, self.prefix))
        self.model = self.create_model()

        # Callbacks
        cbk = GANMonitor(output_dir=os.path.join(self.output_dir, self.prefix), num_img=9, latent_dim=self.n_z, output_epochs=20)
        csv_logger = keras.callbacks.CSVLogger(os.path.join(self.model_dir, self.prefix, 'training_log.csv'), append=True)

        # Start training
        self.model.fit(self.train_images, batch_size=self.batch_size, epochs=self.epochs, callbacks=[cbk, csv_logger])
        self.model.save(os.path.join(self.model_dir, self.prefix, 'model.keras'))
        return self.model

    def simulate_masks(self,
                       no_of_images=1,
                       min_no_of_particles=100,
                       max_no_of_particles=150,
                       use_normal_distribution=False,
                       sigma=0.10,
                       mu=1.0,
                       min_scaling=0.75,
                       max_scaling=1.25,
                       use_perlin_noise=True,
                       perlin_noise_threshold=0.5,
                       perlin_noise_frequency=4,
                       use_random_rotation='DISABLE',  # DISABLE, RANDOM, PERLIN
                       max_overlap=0.01,  # percentage of particle area
                       grid_type='DISABLE',  # DISABLE, HEXAGONAL, CUBIC
                       grid_spacing_factor=0.125,  # percentage of particle size
                       grid_noise_factor=0.05,  # percentage of particle size
                       img_width=384,
                       img_height=384):

        d = math.ceil(math.sqrt((max_scaling * self.train_images.shape[1])**2 + (max_scaling * self.train_images.shape[2])**2))

        if self.model is None:
            # Load the most recent model
            self.model = keras.models.load_model(os.path.join(self.model_dir, os.listdir(self.model_dir)[-1], 'model.keras'), compile=False)

        if use_normal_distribution:
            min_scaling = mu - 3 * sigma
            max_scaling = mu + 3 * sigma

        if use_perlin_noise or use_random_rotation == 'PERLIN':
            perlin_noise_frequency = perlin_noise_frequency
            perlin_noise_threshold = perlin_noise_threshold  # Range [0, 1]; Higher Values give more clustering

        if max_overlap is not None and grid_type != 'HEXAGONAL' and grid_type != 'CUBIC':
            grid_type = 'HEXAGONAL'

        for i in tqdm(range(0, no_of_images)):
            img = np.zeros((img_height + 3*d, img_width + 3*d), dtype='uint8')
            noise_image = None
            no_of_particles = 0
            if grid_type != 'HEXAGONAL' and grid_type != 'CUBIC':
                no_of_particles = random.randint(min_no_of_particles, max_no_of_particles)

            if use_perlin_noise or use_random_rotation == 'PERLIN':
                opensimplex.random_seed()
                ix, iy = np.arange(0, perlin_noise_frequency, perlin_noise_frequency / (img_width+3*d)), np.arange(0, perlin_noise_frequency, perlin_noise_frequency / (img_height+3*d))
                noise_image = opensimplex.noise2array(iy, ix)
                noise_image -= np.min(noise_image)
                noise_image /= np.max(noise_image)/2
                noise_image = noise_image - 1

            if grid_type == 'HEXAGONAL':
                pos_x = np.zeros(math.ceil((img_height+2*d)/(grid_spacing_factor*self.train_images.shape[1])) * math.ceil((img_width+2*d)/(grid_spacing_factor*self.train_images.shape[2])) + 1, dtype='int32')
                pos_y = np.zeros_like(pos_x)
                n = 0
                for k, y in enumerate(range(0, (img_height+2*d), int(grid_spacing_factor*self.train_images.shape[1]))):
                    for j, x in enumerate(range(0, (img_width+2*d), int(grid_spacing_factor*self.train_images.shape[2]))):
                        if x + k % 2 * int(grid_spacing_factor*self.train_images.shape[2]/2) > (img_width+2*d):
                            break
                        pos_x[n] = x + k % 2 * int(grid_spacing_factor*self.train_images.shape[2]/2)
                        pos_y[n] = y
                        n += 1
                pos_x += np.random.randint(int(-grid_noise_factor*self.train_images.shape[2]), int(grid_noise_factor*self.train_images.shape[2]), pos_x.size)
                pos_y += np.random.randint(int(-grid_noise_factor*self.train_images.shape[1]), int(grid_noise_factor*self.train_images.shape[1]), pos_y.size)
                pos_x = np.where(pos_x < 0, 0, pos_x)
                pos_x = np.where(pos_x > img_width+2*d, img_width+2*d, pos_x)
                pos_y = np.where(pos_y < 0, 0, pos_y)
                pos_y = np.where(pos_y > img_height+2*d, img_height+2*d, pos_y)
                if use_perlin_noise:
                    pos_x, pos_y = zip(*[(pos_x[i], pos_y[i]) for i in range(0, len(pos_x)) if noise_image[pos_x[i], pos_y[i]] > (2 * perlin_noise_threshold - 1)])
                no_of_particles = len(pos_x)
            elif grid_type == 'CUBIC':
                pos_y, pos_x = np.mgrid[0:(img_height+2*d):int(grid_spacing_factor*self.train_images.shape[1]), 0:(img_width+2*d):int(grid_spacing_factor*self.train_images.shape[1])]
                pos_x = pos_x.flatten()
                pos_y = pos_y.flatten()
                pos_x += np.random.randint(int(-grid_noise_factor*self.train_images.shape[2]), int(grid_noise_factor*self.train_images.shape[2]), pos_x.size)
                pos_y += np.random.randint(int(-grid_noise_factor*self.train_images.shape[1]), int(grid_noise_factor*self.train_images.shape[1]), pos_y.size)
                pos_x = np.where(pos_x < 0, 0, pos_x)
                pos_x = np.where(pos_x > img_width+2*d, img_width+2*d, pos_x)
                pos_y = np.where(pos_y < 0, 0, pos_y)
                pos_y = np.where(pos_y > img_height+2*d, img_height+2*d, pos_y)
                if use_perlin_noise:
                    pos_x, pos_y = zip(*[(pos_x[i], pos_y[i]) for i in range(0, len(pos_x)) if noise_image[pos_x[i], pos_y[i]] > (2 * perlin_noise_threshold - 1)])
                no_of_particles = len(pos_x)
            elif use_perlin_noise:
                # pos_values = np.where(noise_image > 0, noise_image, 0).ravel()
                pos_values = (noise_image > (2 * perlin_noise_threshold - 1)).astype('float32') * 1.0  # remap noise_threshold to range [-1, 1] and use thresholded image (all positions > threshold are equally likely)
                pos_values /= np.sum(pos_values)
                selection = np.random.choice(noise_image.ravel(), no_of_particles, replace=False, p=pos_values.ravel())
                selection = np.asarray(np.nonzero(np.isin(noise_image, selection))).transpose()  # order does not matter, so just reshuffle
                np.random.shuffle(selection)
                pos_x, pos_y = selection.transpose()
            else:
                pos_x, pos_y = np.random.randint(0, (img_width+2*d), no_of_particles), np.random.randint(0, (img_height+2*d), no_of_particles)

            if use_normal_distribution:
                scalings = np.random.normal(mu, sigma, no_of_particles)
            else:
                scalings = np.random.uniform(min_scaling, max_scaling, no_of_particles)
            scalings = np.where(scalings > max_scaling, max_scaling, scalings)
            scalings = np.where(scalings < min_scaling, min_scaling, scalings)

            if use_random_rotation == 'RANDOM':
                rotations = np.random.randint(0, 360, no_of_particles)
            elif use_random_rotation == 'PERLIN':
                rotations = noise_image[pos_y, pos_x] * 180  # noise image has range [-1, 1]
            else:
                rotations = np.zeros(no_of_particles)

            samples = np.zeros((no_of_particles, self.train_images.shape[1], self.train_images.shape[2], 1), dtype='float32')
            if no_of_particles > self.batch_size:
                for j in range(0, no_of_particles-self.batch_size, self.batch_size):
                    random_latent_vectors = keras.random.normal(shape=(self.batch_size, self.n_z))
                    out = WGAN_GP.to_numpy_array(self.model(random_latent_vectors, training=False))
                    samples[j:j+self.batch_size] = out
            
            if no_of_particles % self.batch_size == 0:
                random_latent_vectors = keras.random.normal(shape=(self.batch_size, self.n_z))
                samples[-self.batch_size:] = WGAN_GP.to_numpy_array(self.model(random_latent_vectors, training=False))
            else:
                random_latent_vectors = keras.random.normal(shape=(no_of_particles % self.batch_size, self.n_z))
                samples[-(no_of_particles % self.batch_size):] = WGAN_GP.to_numpy_array(self.model(random_latent_vectors, training=False))

            samples = (samples * 127.5 + 127.5)[:, :, :, 0].astype('uint8')

            for j, p in enumerate(samples):
                height, width = p.shape
                image_center = (width / 2, height / 2)

                rotation_mat = cv2.getRotationMatrix2D(image_center, rotations[j], scalings[j])

                abs_cos = abs(rotation_mat[0, 0])
                abs_sin = abs(rotation_mat[0, 1])

                bound_w = int(width * abs_sin + height * abs_cos)
                bound_h = int(width * abs_cos + height * abs_sin)

                rotation_mat[0, 2] += bound_h / 2 - image_center[0]
                rotation_mat[1, 2] += bound_w / 2 - image_center[1]

                p = cv2.warpAffine(p, rotation_mat, (bound_h, bound_w))
                p = p > 127
                p = ndimage.binary_fill_holes(p)
                p = ndimage.binary_opening(p, structure=np.ones((9, 9)))

                p_eroded = ndimage.binary_erosion(p, iterations=2)
                if np.any(p_eroded > 0):
                    if max_overlap is not None and np.sum(np.logical_and(img[pos_y[j]:pos_y[j] + p.shape[0], pos_x[j]:pos_x[j] + p.shape[1]], p_eroded).astype('int32')) > max_overlap*np.sum(p_eroded.astype('uint8')):
                        continue
                    img[pos_y[j]:pos_y[j] + p.shape[0], pos_x[j]:pos_x[j] + p.shape[1]] -= np.logical_and(img[pos_y[j]:pos_y[j] + p.shape[0], pos_x[j]:pos_x[j] + p.shape[1]], p)
                    img[pos_y[j]:pos_y[j] + p.shape[0], pos_x[j]:pos_x[j] + p.shape[1]] += p_eroded

            a = int((img.shape[0]-img_height)/2)
            b = int((img.shape[1]-img_width)/2)
            img = img[a:a+img_height, b:b+img_width] * 255
            img = Image.fromarray(img)
            img.save(os.path.join(self.generate_dir, '{:05d}.tif'.format(i)))

        # Choose 5 random files for testing
        input_files = [f for f in os.listdir(self.generate_dir) if '.tif' in f or '.png' in f or '.bmp' in f]
        test_img = random.sample(input_files, 5)
        input_dir = self.generate_dir
        output_dir = os.path.join(self.generate_dir, '..', 'testB')
        for f in test_img:
            copy(os.path.join(input_dir, f), output_dir)

    ########################
    # Network Architecture #
    ########################

    @staticmethod
    def conv_block(
        x,
        filters,
        activation,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        use_bias=True,
        use_bn=False,
        use_dropout=False,
        drop_value=0.5,
    ):
        x = keras.layers.Conv2D(
            filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias
        )(x)
        if use_bn:
            x = keras.layers.BatchNormalization()(x)
        x = activation(x)
        if use_dropout:
            x = keras.layers.Dropout(drop_value)(x)
        return x

    def get_discriminator_model(self):
        img_input = keras.layers.Input(shape=self.train_images.shape[1:])
        x = self.conv_block(
            img_input,
            64,
            kernel_size=(5, 5),
            strides=(2, 2),
            use_bn=False,
            use_bias=True,
            activation=keras.layers.LeakyReLU(0.2),
            use_dropout=False,
            drop_value=0.3,
        )
        x = self.conv_block(
            x,
            128,
            kernel_size=(5, 5),
            strides=(2, 2),
            use_bn=False,
            activation=keras.layers.LeakyReLU(0.2),
            use_bias=True,
            use_dropout=True,
            drop_value=0.3,
        )
        x = self.conv_block(
            x,
            256,
            kernel_size=(5, 5),
            strides=(2, 2),
            use_bn=False,
            activation=keras.layers.LeakyReLU(0.2),
            use_bias=True,
            use_dropout=True,
            drop_value=0.3,
        )
        x = self.conv_block(
            x,
            512,
            kernel_size=(5, 5),
            strides=(2, 2),
            use_bn=False,
            activation=keras.layers.LeakyReLU(0.2),
            use_bias=True,
            use_dropout=False,
            drop_value=0.3,
        )

        x = keras.layers.Flatten()(x)
        x = keras.layers.Dropout(0.2)(x)
        x = keras.layers.Dense(1)(x)

        d_model = keras.models.Model(img_input, x, name="discriminator")
        return d_model

    @staticmethod
    def upsample_block(
        x,
        filters,
        activation,
        kernel_size=(3, 3),
        strides=(1, 1),
        up_size=(2, 2),
        padding="same",
        use_bn=False,
        use_bias=True,
        use_dropout=False,
        drop_value=0.3,
    ):
        x = keras.layers.UpSampling2D(up_size)(x)
        x = keras.layers.Conv2D(
            filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias
        )(x)

        if use_bn:
            x = keras.layers.BatchNormalization()(x)

        if activation:
            x = activation(x)
        if use_dropout:
            x = keras.layers.Dropout(drop_value)(x)
        return x

    def get_generator_model(self):
        noise = keras.layers.Input(shape=(self.n_z,))
        x = keras.layers.Dense((self.train_images.shape[1])//8 * (self.train_images.shape[2])//8 * 4 * 8 * 8, use_bias=False)(noise)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.LeakyReLU(0.2)(x)

        x = keras.layers.Reshape(((self.train_images.shape[1])//8, (self.train_images.shape[2])//8, 256))(x)
        x = self.upsample_block(
            x,
            128,
            keras.layers.LeakyReLU(0.2),
            strides=(1, 1),
            use_bias=False,
            use_bn=True,
            padding="same",
            use_dropout=False,
        )
        x = self.upsample_block(
            x,
            64,
            keras.layers.LeakyReLU(0.2),
            strides=(1, 1),
            use_bias=False,
            use_bn=True,
            padding="same",
            use_dropout=False,
        )
        x = self.upsample_block(
            x, 1, keras.layers.Activation("tanh"), strides=(1, 1), use_bias=False, use_bn=True
        )

        g_model = keras.models.Model(noise, x, name="generator")
        return g_model

    # Create Model
    # Define the loss functions to be used for discriminator
    # This should be (fake_loss - real_loss)
    # We will add the gradient penalty later to this loss function
    @staticmethod
    def discriminator_loss(real_img, fake_img):
        real_loss = keras.ops.mean(real_img)
        fake_loss = keras.ops.mean(fake_img)
        return fake_loss - real_loss

    # Define the loss functions to be used for generator
    @staticmethod
    def generator_loss(fake_img):
        return -keras.ops.mean(fake_img)

    def create_model(self):
        # Optimizer for both the networks
        # learning_rate=0.0002, beta_1=0.5 are recommended
        generator_optimizer = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)
        discriminator_optimizer = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)
        # Get the wgan model
        model = WGAN_GP(
            discriminator=self.get_discriminator_model(),
            generator=self.get_generator_model(),
            latent_dim=self.n_z,
            discriminator_extra_steps=3,
        )

        # Compile the WGAN model
        model.compile(
            d_optimizer=discriminator_optimizer,
            g_optimizer=generator_optimizer,
            g_loss_fn=self.generator_loss,
            d_loss_fn=self.discriminator_loss,
        )
        return model


if __name__ == '__main__':
    WGAN = WGAN(root_dir='./')
