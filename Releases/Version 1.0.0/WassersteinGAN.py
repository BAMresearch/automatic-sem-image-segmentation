import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm.autonotebook import tqdm
from opensimplex import OpenSimplex

import os
import imageio
import random
import cv2
import math
from scipy import ndimage

import argparse

class WGAN_GP(tf.keras.Model):
    def __init__(
            self,
            discriminator,
            generator,
            latent_dim,
            discriminator_extra_steps=3,
            gp_weight=10.0,
    ):
        super(WGAN_GP, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.d_steps = discriminator_extra_steps
        self.gp_weight = gp_weight

    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        super(WGAN_GP, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

    def gradient_penalty(self, batch_size, real_images, fake_images):
        """ Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # get the interplated image
        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.discriminator(interpolated, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calcuate the norm of the gradients
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, real_images):
        if isinstance(real_images, tuple):
            real_images = real_images[0]

        # Get the batch size
        batch_size = tf.shape(real_images)[0]

        # For each batch, we are going to perform the
        # following steps as laid out in the original paper.
        # 1. Train the generator and get the generator loss
        # 2. Train the discriminator and get the discriminator loss
        # 3. Calculate the gradient penalty
        # 4. Multiply this gradient penalty with a constant weight factor
        # 5. Add gradient penalty to the discriminator loss
        # 6. Return generator and discriminator losses as a loss dictionary.

        # Train discriminator first. The original paper recommends training
        # the discriminator for `x` more steps (typically 5) as compared to
        # one step of the generator. Here we will train it for 3 extra steps
        # as compared to 5 to reduce the training time.
        for i in range(self.d_steps):
            # Get the latent vector
            random_latent_vectors = tf.random.normal(
                shape=(batch_size, self.latent_dim)
            )
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
                gp = self.gradient_penalty(batch_size, real_images, fake_images)
                # Add the gradient penalty to the original discriminator loss
                d_loss = d_cost + gp * self.gp_weight

            # Get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            # Update the weights of the discriminator using the discriminator optimizer
            self.d_optimizer.apply_gradients(
                zip(d_gradient, self.discriminator.trainable_variables)
            )

        # Train the generator now.
        # Get the latent vector
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        with tf.GradientTape() as tape:
            # Generate fake images using the generator
            generated_images = self.generator(random_latent_vectors, training=True)
            # Get the discriminator logits for fake images
            gen_img_logits = self.discriminator(generated_images, training=True)
            # Calculate the generator loss
            g_loss = self.g_loss_fn(gen_img_logits)

        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )
        return {"d_loss": d_loss, "g_loss": g_loss}

    def generate(self, z):
        return self.gen(z)

    def discriminate(self, x):
        return self.disc(x)


class GANMonitor(tf.keras.callbacks.Callback):
    def __init__(self, outputDir, num_img=9, latent_dim=128, outputEpochs=100):
        super(GANMonitor, self).__init__()
        self.num_img = num_img
        self.latent_dim = latent_dim
        self.EPOCHS = outputEpochs
        self.OUTPUT_DIR = outputDir

    def on_epoch_end(self, epoch, logs=None):
        if epoch % int(self.EPOCHS) == 0:
            self.plot_reconstruction(self.model, epoch, nex=self.num_img)

    def plot_reconstruction(self, model, epoch, nex=9, zm=2):
        random_latent_vectors = tf.random.normal(shape=(nex, self.latent_dim))
        samples = model.generator(random_latent_vectors).numpy()
        cols = 3
        rows = math.ceil(nex/float(cols))
        fig = plt.figure(figsize=(2, 2 * rows / cols))
        for i, s in enumerate(samples):
            fig.add_subplot(rows, cols, i + 1)
            plt.axis('off')
            plt.imshow((s*127.5+127.5)[:, :, 0].astype('uint8'), cmap="gray")

        plt.savefig(os.path.join(self.OUTPUT_DIR, 'Epoch_{:05d}'.format(epoch)))
        plt.close(fig)


class WGAN():
    def __init__(self, ROOT_DIR, instance_image_shape=(64, 64, 1), image_shape=(352, 512, 1)):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

        self.ROOT_DIR = os.path.join(ROOT_DIR, '1_WGAN')
        self.INPUT_DIR = os.path.join(ROOT_DIR, 'Input_Masks')
        self.OUTPUT_DIR = os.path.join(self.ROOT_DIR, 'Output_Images')
        self.MODEL_DIR = os.path.join(self.ROOT_DIR, 'Models')
        self.GENERATE_DIR = os.path.join(self.ROOT_DIR, 'Generated_Masks')

        self.BATCH_SIZE = 64
        self.DIMS = instance_image_shape
        self.EPOCHS = 1000
        self.N_Z = 128

        self.INSTANCE_IMAGE_SHAPE = instance_image_shape
        self.IMAGE_SHAPE = image_shape

        # Create dataset
        # load dataset
        self.inputFiles = []
        self.train_images = []

        for file in os.listdir(self.INPUT_DIR):
            if ".tif" in file:
                self.inputFiles.append(os.path.join(self.INPUT_DIR, file))


        # For implementation from https://keras.io/examples/generative/wgan_gp/

        for f in self.inputFiles:
            image = np.asarray(imageio.imread(f), dtype='float32').copy()

            self.train_images.append(image.copy())
            self.train_images.append(np.fliplr(image.copy()))
            self.train_images.append(np.flipud(image.copy()))
            self.train_images.append(np.flipud(np.fliplr(image.copy())))

        self.train_images = np.asarray(self.train_images)
        self.train_images = (self.train_images.reshape((self.train_images.shape[0], 64, 64, 1)).astype("float32") - 127.5) / 127.5

    def conv_block(
        self,
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
        x = tf.keras.layers.Conv2D(
            filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias
        )(x)
        if use_bn:
            x = tf.keras.layers.BatchNormalization()(x)
        x = activation(x)
        if use_dropout:
            x = tf.keras.layers.Dropout(drop_value)(x)
        return x


    def get_discriminator_model(self):
        img_input = tf.keras.layers.Input(shape=self.DIMS)
        # Zero pad the input to make the input images size to (32, 32, 1).
        # x = tf.keras.layers.ZeroPadding2D((2, 2))(img_input)
        x = self.conv_block(
            img_input,
            64,
            kernel_size=(5, 5),
            strides=(2, 2),
            use_bn=False,
            use_bias=True,
            activation=tf.keras.layers.LeakyReLU(0.2),
            use_dropout=False,
            drop_value=0.3,
        )
        x = self.conv_block(
            x,
            128,
            kernel_size=(5, 5),
            strides=(2, 2),
            use_bn=False,
            activation=tf.keras.layers.LeakyReLU(0.2),
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
            activation=tf.keras.layers.LeakyReLU(0.2),
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
            activation=tf.keras.layers.LeakyReLU(0.2),
            use_bias=True,
            use_dropout=False,
            drop_value=0.3,
        )

        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(1)(x)

        d_model = tf.keras.models.Model(img_input, x, name="discriminator")
        return d_model


    def upsample_block(
        self,
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
        x = tf.keras.layers.UpSampling2D(up_size)(x)
        x = tf.keras.layers.Conv2D(
            filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias
        )(x)

        if use_bn:
            x = tf.keras.layers.BatchNormalization()(x)

        if activation:
            x = activation(x)
        if use_dropout:
            x = tf.keras.layers.Dropout(drop_value)(x)
        return x

    def get_generator_model(self):
        noise = tf.keras.layers.Input(shape=(self.N_Z,))
        x = tf.keras.layers.Dense(8 * 8 * 256, use_bias=False)(noise)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)

        x = tf.keras.layers.Reshape((8, 8, 256))(x)
        x = self.upsample_block(
            x,
            128,
            tf.keras.layers.LeakyReLU(0.2),
            strides=(1, 1),
            use_bias=False,
            use_bn=True,
            padding="same",
            use_dropout=False,
        )
        x = self.upsample_block(
            x,
            64,
            tf.keras.layers.LeakyReLU(0.2),
            strides=(1, 1),
            use_bias=False,
            use_bn=True,
            padding="same",
            use_dropout=False,
        )
        x = self.upsample_block(
            x, 1, tf.keras.layers.Activation("tanh"), strides=(1, 1), use_bias=False, use_bn=True
        )
        # At this point, we have an output which has the same shape as the input, (32, 32, 1).
        # We will use a Cropping2D layer to make it (28, 28, 1).
        # x = tf.keras.layers.Cropping2D((2, 2))(x)

        g_model = tf.keras.models.Model(noise, x, name="generator")
        return g_model


    # Create Model
    # Define the loss functions to be used for discrimiator
    # This should be (fake_loss - real_loss)
    # We will add the gradient penalty later to this loss function
    def discriminator_loss(self, real_img, fake_img):
        real_loss = tf.reduce_mean(real_img)
        fake_loss = tf.reduce_mean(fake_img)
        return fake_loss - real_loss

    # Define the loss functions to be used for generator
    def generator_loss(self, fake_img):
        return -tf.reduce_mean(fake_img)

    def createModel(self):
        # Optimizer for both the networks
        # learning_rate=0.0002, beta_1=0.5 are recommended
        generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)
        discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)
        # Get the wgan model
        model = WGAN_GP(
            discriminator=self.get_discriminator_model(),
            generator=self.get_generator_model(),
            latent_dim=self.N_Z,
            discriminator_extra_steps=3,
        )

        # Compile the wgan model
        model.compile(
            d_optimizer=discriminator_optimizer,
            g_optimizer=generator_optimizer,
            g_loss_fn=self.generator_loss,
            d_loss_fn=self.discriminator_loss,
        )
        return model

    def eightToFourConnected(self, img):
        for x in range(0, img.shape[0]-1):
            for y in range(0, img.shape[1]-1):
                if img[x, y] == 0 and img[x + 1, y + 1] == 0 and img[x + 1, y] != 0 and img[x, y + 1] != 0:
                    img[x + 1, y] = 0
                elif img[x + 1, y] == 0 and img[x, y + 1] == 0 and img[x, y] != 0 and img[x + 1, y + 1] != 0:
                    img[x, y] = 0
        return img

    def startTraining(self):
        prefix = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())

        model = self.createModel()

        # Callbacks
        cbk = GANMonitor(outputDir=self.OUTPUT_DIR, num_img=9, latent_dim=self.N_Z)
        csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(self.MODEL_DIR, prefix + '_training_log.csv'), append=True)

        # Start training
        model.fit(self.train_images, batch_size=self.BATCH_SIZE, epochs=self.EPOCHS, callbacks=[cbk, csv_logger])
        model.save_weights(os.path.join(self.MODEL_DIR, prefix + '_Weights.h5'))
        return model

    def simulateMasks(self, model, NO_OF_IMAGES=1000, MIN_NO_OF_PARTICLES=100, MAX_NO_OF_PARTICLES=150, SIGMA=0.10, MU=1.0, USE_PERLIN_NOISE=False, USE_NORMAL_DISTRIBUTION=False):
        IMG_WIDTH = self.IMAGE_SHAPE[1]
        IMG_HEIGHT = self.IMAGE_SHAPE[0]
        if USE_NORMAL_DISTRIBUTION:
            MIN_SCALING = MU - 3*SIGMA
            MAX_SCALING = MU + 3*SIGMA
        else:
            MIN_SCALING = 0.5
            MAX_SCALING = 1.5

        if USE_PERLIN_NOISE:
            OCTAVES = 1.0
            FREQ = IMG_WIDTH//4
            NOISE_THRESHOLD = 0.5  # Higher Values give more clustering

        for i in tqdm(range(0, NO_OF_IMAGES)):
            random_latent_vectors = tf.random.normal(shape=(random.randrange(MIN_NO_OF_PARTICLES, MAX_NO_OF_PARTICLES), self.N_Z))
            samples = model.generator(random_latent_vectors).numpy() * 127.5 + 127.5
            img = np.zeros((IMG_HEIGHT + int(MAX_SCALING * self.DIMS[0]), IMG_WIDTH + int(MAX_SCALING * self.DIMS[1])), dtype='uint8')

            if USE_PERLIN_NOISE:
                FREQ *= OCTAVES
                simplexNoise = OpenSimplex(seed=random.randint(0, 10*NO_OF_IMAGES))
                noiseImage = np.zeros_like(img, dtype='float32')
                for x in range(0, noiseImage.shape[1]):
                    for y in range(0, noiseImage.shape[0]):
                        noiseImage[y, x] = simplexNoise.noise2d(x / FREQ, y / FREQ)
                noiseImage -= np.min(noiseImage)
                noiseImage /= np.max(noiseImage)
                noiseImage = (noiseImage >= NOISE_THRESHOLD)

            for p in samples:
                if USE_NORMAL_DISTRIBUTION:
                    scaling = random.gauss(MU, SIGMA)
                else:
                    scaling = random.uniform(MIN_SCALING, MAX_SCALING)

                scaling = max(scaling, MIN_SCALING)
                scaling = min(scaling, MAX_SCALING)
                p = cv2.resize(p, dsize=(int(scaling * p.shape[0]), int(scaling * p.shape[1])), interpolation=cv2.INTER_CUBIC)
                p = p > 127
                p = ndimage.binary_fill_holes(p)
                p = ndimage.binary_opening(p, structure=np.ones((9, 9)))

                if np.any(ndimage.binary_erosion(p, iterations=2) > 0):
                    posX = random.randrange(0, IMG_WIDTH)
                    posY = random.randrange(0, IMG_HEIGHT)

                    if USE_PERLIN_NOISE:
                        while not noiseImage[posY + p.shape[0]//2, posX + p.shape[1]//2]:
                            posX = random.randrange(0, IMG_WIDTH)
                            posY = random.randrange(0, IMG_HEIGHT)

                    img[posY:posY + p.shape[0], posX:posX + p.shape[1]] -= np.logical_and(img[posY:posY+p.shape[0], posX:posX+p.shape[1]], p)
                    img[posY:posY + p.shape[0], posX:posX + p.shape[1]] += ndimage.binary_erosion(p, iterations=2)

            a = int((img.shape[0]-IMG_HEIGHT)/2)
            b = int((img.shape[1]-IMG_WIDTH)/2)
            img = img[a:a+IMG_HEIGHT, b:b+IMG_WIDTH] * 255
            imageio.imwrite(os.path.join(self.GENERATE_DIR, '{:05d}.tif'.format(i)), img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Implementation of the Wasserstein GAN with gradient penalty.')
    parser.add_argument('root_dir', type=str, help='Root directory where the folders with the images and masks are stored.')
    parser.add_argument('--instance_image_shape', type=str, default="(64, 64, 1)", help='Image shape of individual particle instances.')
    parser.add_argument('--image_shape', type=str, default="(352, 512, 1)", help='Image shape of simulated segmentation masks.')
    parser.add_argument('--image_number', type=int, default=1000, help='Number of simulated segmentation masks to be created.')

    args = parser.parse_args()
    ROOT_DIR = args.root_dir
    instance_image_shape = [int(a) for a in args.instance_image_shape.strip(')(').replace(' ', '').split(',')]
    image_shape = [int(a) for a in args.image_shape.strip(')(').replace(' ', '').split(',')]

    WGAN = WGAN(ROOT_DIR=ROOT_DIR, instance_image_shape=(64, 64, 1), image_shape=(352, 512, 1))
    trainedWGANModel = WGAN.startTraining()
    WGAN.simulateMasks(trainedWGANModel, NO_OF_IMAGES=args.image_number, USE_PERLIN_NOISE=True, USE_NORMAL_DISTRIBUTION=True)

