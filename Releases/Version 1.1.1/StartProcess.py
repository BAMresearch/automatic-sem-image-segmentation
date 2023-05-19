import os
from datetime import datetime
import multiprocessing as mp

import WassersteinGAN
import CycleGAN
import UNet_Segmentation
import HelperFunctions

# General Setup
ROOT_DIR = os.path.abspath("./")                                            # Root Directory for process
INPUT_DIR_MASKS = os.path.join(ROOT_DIR, 'Input_Masks')                     # Directory with exemplary masks of single particles
INPUT_DIR_IMAGES = os.path.join(ROOT_DIR, 'Input_Images')                   # Directory with the real images that should be segmented
OUTPUT_DIR_CYCLEGAN = os.path.join(ROOT_DIR, 'Output_Masks_CycleGAN')       # Output directory for the image segmentations produced by CycleGAN
OUTPUT_DIR_UNET = os.path.join(ROOT_DIR, 'Output_Masks_UNet')               # Output directory for the image segmentations produced by UNet
TILE_SIZE_W = 384                                                           # Tile width (images will be tiled since training on the full image is usually not possible due to GPU memory limitations)
TILE_SIZE_H = 384                                                           # Tile height (images will be tiled since training on the full image is usually not possible due to GPU memory limitations)
NUM_SIMULATED_MASKS = 1000                                                  # Minimum number of masks to simulate by WGAN (if there are more "real" images available, more masks will be synthesized, if there rae more simulated masks, real images will be augmented by applying random cropping and mirroring)
RUN_INFERENCE_ON_WHOLE_IMAGE = True                                         # Run inference on whole images instead of image tiles (can help reduce tiling artifacts)
DARK_BACKGROUND = True                                                      # Whether the particles are bright on a dark background (e.g., in SEM or HAADF-STEM) or dark on a bright background (e.g., in Brightfield TEM)

# Options for training and inference on a GPU
USE_GPUS_NO = (0, )                                                         # List of GPUs used for training (if there is more than one available)
USE_GPU_FOR_WHOLE_IMAGE_INFERENCE = False                                   # If set to False, inference of whole images (as opposed to image tiles) will be done on a CPU (slower, but generally necessary due to GPU memory restrictions). Has no effect if RUN_INFERENCE_ON_WHOLE_IMAGE=False
ALLOW_MEMORY_GROWTH = True                                                  # Whether to pre-allocate all memory at the beginning or allow for memory growth

# Options for Training the Networks - for more options see calls to in the individual steps and the parameters in the constructors/methods of the files WassersteinGAN.py, CycleGAN.py, and UNet_Segmentation.py
WGAN_BATCH_SIZE = 64                                                        # Batch size used during WassersteinGAN training
WGAN_EPOCHS = 1000                                                          # Number of training epochs for WassersteinGAN
CYCLEGAN_BATCH_SIZE = 2                                                     # Batch size used during CycleGAN training
CYCLEGAN_EPOCHS = 50                                                        # Number of training epochs for CycleGAN
CYCLEGAN_USE_SKIPS = True                                                   # Use skip connections in CycleGAN (conceptually somewhat similar to identity mappings)
GAUSSIAN_BLUR_AMOUNT = 0.0                                                  # Gaussian Blur added to the generated fake SEM images (can help with checkerboard artifacts or too strong outlines when skip connections/identity mappings are used)
UNET_BATCH_SIZE = 2                                                         # Batch size used during UNet training
UNET_EPOCHS = 50                                                            # Number of training epochs for UNet
UNET_CONTRAST_OPTIMIZATION_RANGE = (0.5, 99.5)                              # Remove "hot" and "cold" pixels by normalizing the contrast range to lie between the two specified percentiles
UNET_FILTERS = 16                                                           # Number of filters in the first UNet layer
USE_DATALOADER = True                                                       # Use a dataloader for training CycleGAN and UNet (enable for very large training sets that cannot be loaded into available CPU memory at once)

################################################################################################################
# Wrapper functions for starting subprocesses (workaround for problems with tensorflow not freeing GPU memory) #
################################################################################################################

use_gpu_for_inference = not RUN_INFERENCE_ON_WHOLE_IMAGE or (USE_GPU_FOR_WHOLE_IMAGE_INFERENCE and RUN_INFERENCE_ON_WHOLE_IMAGE)


def start_step_0():
    print('Step0: Configuring Devices, Initializing Directories, and Preparing Images...')
    HelperFunctions.initialize_directories(root_dir=ROOT_DIR, output_dir_cyclegan=OUTPUT_DIR_CYCLEGAN, output_dir_unet=OUTPUT_DIR_UNET)
    HelperFunctions.prepare_images_cycle_gan(root_dir=ROOT_DIR, input_dir_images=INPUT_DIR_IMAGES, tile_size_w=TILE_SIZE_W, tile_size_h=TILE_SIZE_H, num_simulated_masks=NUM_SIMULATED_MASKS, dark_background=DARK_BACKGROUND)


def start_step_1():
    print('Step 1: Training WGAN...')
    wgan = WassersteinGAN.WGAN(root_dir=ROOT_DIR, allow_memory_growth=ALLOW_MEMORY_GROWTH, use_gpus_no=USE_GPUS_NO)
    wgan.batch_size = WGAN_BATCH_SIZE                       # Batch size during training
    wgan.epochs = WGAN_EPOCHS                               # Training epochs
    wgan.n_z = 128                                          # Noise vector size
    wgan.start_training()


def start_step_2():
    print('Step 2: Simulating fake masks...')
    num_masks = max(NUM_SIMULATED_MASKS, len(os.listdir(os.path.join(ROOT_DIR, '2_CycleGAN', 'data', 'trainA'))))
    w_gan = WassersteinGAN.WGAN(root_dir=ROOT_DIR, allow_memory_growth=ALLOW_MEMORY_GROWTH, use_gpus_no=USE_GPUS_NO)
    w_gan.n_z = 128                                         # Noise vector size
    w_gan.simulate_masks(no_of_images=num_masks,            # No of fake masks to simulate
                         min_no_of_particles=100,           # Minimum number of particles per image tile (does not take overlaps into account)
                         max_no_of_particles=150,           # Maximum number of particles per image tile (does not take overlaps into account)
                         use_perlin_noise=True,             # Use Perlin Noise to simulate particle agglomeration/aggregation
                         perlin_noise_threshold=0.5,        # Threshold for Perlin Noise - higher values give smaller patches with more particle aggregation
                         perlin_noise_frequency=4,          # Determines the size and number of patches (higher values give more but smaller patches)
                         use_normal_distribution=True,      # Use a normal distribution to adjust particle size
                         use_random_rotation='DISABLE',     # 'DISABLE': Do not apply any additional rotation; 'RANDOM': Apply additional random rotation; 'PERLIN': Apply 'continuous', spatially correlated random rotations
                         grid_type='DISABLE',               # Arrange particles in a grid (DISABLE, HEXAGONAL, CUBIC)
                         max_overlap=0.01,                  # Percentage of particle area
                         img_width=TILE_SIZE_W,             # Width of the simulated images
                         img_height=TILE_SIZE_H)            # Height of the simulated images


def start_step_3():
    print('Step 3: Training CycleGAN...')
    cycle_gan = CycleGAN.CycleGAN(root_dir=ROOT_DIR, image_shape=(TILE_SIZE_H, TILE_SIZE_W, 1), allow_memory_growth=ALLOW_MEMORY_GROWTH, use_gpus_no=USE_GPUS_NO)
    cycle_gan.batch_size = CYCLEGAN_BATCH_SIZE              # Batch size during training
    cycle_gan.epochs = CYCLEGAN_EPOCHS                      # Training epochs
    cycle_gan.use_data_loader = USE_DATALOADER              # Whether to use a dataloader
    cycle_gan.label_smoothing_factor = 0.0                  # Label smoothing factor - set to a small value (e.g., 0.1) to avoid overconfident discriminator guesses and very low discriminator losses (too strong discriminators can be problematic for generators due to the adversarial nature of GANs)
    cycle_gan.gaussian_noise_value = 0.15                   # Set to a small value (e.g., 0.15) to add Gaussian Noise to the discriminator layers (can help against mode collapse and "overtraining" the discriminator)
    cycle_gan.use_skip_connection = CYCLEGAN_USE_SKIPS      # Add a skip connection between the input and output layer in the generator (conceptually similar to identity mappings)
    cycle_gan.lambda_identity_a = 0.5
    cycle_gan.lambda_identity_b = 0.5
    cycle_gan.filters = 64
    cycle_gan.use_binary_crossentropy = False
    cycle_gan.start_training()


def start_step_4():
    print('Step 4: Generating fake training images and segmenting real images with CycleGAN...')
    # Generate fake images for training from simulated masks
    cycle_gan = CycleGAN.CycleGAN(root_dir=ROOT_DIR, image_shape=(TILE_SIZE_H, TILE_SIZE_W, 1), allow_memory_growth=ALLOW_MEMORY_GROWTH, use_gpus_no=USE_GPUS_NO)
    cycle_gan.use_skip_connection = CYCLEGAN_USE_SKIPS      # Add a skip connection between the input and output layer in the generator (conceptually similar to identity mappings)
    cycle_gan.filters = 64
    cycle_gan.use_binary_crossentropy = False
    cycle_gan.run_inference(files=os.path.join(ROOT_DIR, '2_CycleGAN', 'data', 'trainB'),
                            output_directory=os.path.join(ROOT_DIR, '2_CycleGAN', 'generate_images', 'A'),
                            source_domain='B',
                            tile_images=False,
                            use_gpu=True)

    # Segment real images with CycleGAN
    cycle_gan.image_shape = (TILE_SIZE_W, TILE_SIZE_H)      # Size of tiles (when not running inference on the whole image)
    cycle_gan.run_inference(files=INPUT_DIR_IMAGES,
                            output_directory=os.path.join(ROOT_DIR, '2_CycleGAN', 'generate_images', 'B'),
                            source_domain='A',
                            tile_images=not RUN_INFERENCE_ON_WHOLE_IMAGE,
                            min_overlap=2,
                            manage_overlap_mode=2,
                            use_gpu=use_gpu_for_inference)


def start_step_5():
    print('Step 5: Postprocessing CycleGAN Output images...')
    HelperFunctions.filter_gan_masks(img_path=os.path.join(ROOT_DIR, '2_CycleGAN', 'generate_images', 'A'),
                                     msk_path=os.path.join(ROOT_DIR, '2_CycleGAN', 'data', 'trainB'),
                                     out_path=os.path.join(ROOT_DIR, '2_CycleGAN', 'generate_images', 'Synthetic_Masks_Filtered'),
                                     gaussian_blur_amount=GAUSSIAN_BLUR_AMOUNT,
                                     do_watershed_and_four_connectivity=False,
                                     dark_background=DARK_BACKGROUND)

    HelperFunctions.filter_gan_masks(img_path=INPUT_DIR_IMAGES,
                                     msk_path=os.path.join(ROOT_DIR, '2_CycleGAN', 'generate_images', 'B'),
                                     out_path=OUTPUT_DIR_CYCLEGAN,
                                     do_watershed_and_four_connectivity=True,
                                     dark_background=DARK_BACKGROUND)


def start_step_6a():
    print('Step 6.a: Train MultiRes UNet...')
    u_net = UNet_Segmentation.UNet(root_dir=ROOT_DIR, image_dir=os.path.join(ROOT_DIR, '2_CycleGAN', 'generate_images', 'A'), mask_dir=os.path.join(ROOT_DIR, '2_CycleGAN', 'generate_images', 'Synthetic_Masks_Filtered'), allow_memory_growth=ALLOW_MEMORY_GROWTH, use_gpus_no=USE_GPUS_NO)
    u_net.batch_size = UNET_BATCH_SIZE                                       # Batch size during training
    u_net.epochs = UNET_EPOCHS                                               # Training epochs
    u_net.use_dataloader = USE_DATALOADER                                    # Whether to use a dataloader
    u_net.filters = UNET_FILTERS                                             # Filters in the first layer of the UNet
    u_net.contrast_optimization_range = UNET_CONTRAST_OPTIMIZATION_RANGE     # Contrast optimization range (can be used to remove "hot" and "cold" pixels)
    u_net.run_training()


def start_step_6b():
    print('Segment real images with UNet')
    u_net = UNet_Segmentation.UNet(root_dir=ROOT_DIR, image_dir=os.path.join(ROOT_DIR, '2_CycleGAN', 'generate_images', 'A'), mask_dir=os.path.join(ROOT_DIR, '2_CycleGAN', 'generate_images', 'Synthetic_Masks_Filtered'), allow_memory_growth=ALLOW_MEMORY_GROWTH, use_gpus_no=USE_GPUS_NO)
    u_net.use_dataloader = USE_DATALOADER                                    # Whether to use a dataloader
    u_net.filters = UNET_FILTERS                                             # Filters in the first layer of the UNet
    u_net.image_shape = (TILE_SIZE_W, TILE_SIZE_H)                           # Size of tiles (when not running inference on the whole image)
    u_net.contrast_optimization_range = UNET_CONTRAST_OPTIMIZATION_RANGE     # Contrast optimization range (can be used to remove "hot" and "cold" pixels)
    u_net.run_inference(files=INPUT_DIR_IMAGES,                              # Directory with images to segment
                        output_directory=OUTPUT_DIR_UNET,                    # Output directory for segmented images
                        tile_images=not RUN_INFERENCE_ON_WHOLE_IMAGE,        # Whether to tile images
                        threshold=-1,                                        # Threshold applied to segmentation masks (value between [0, 1]; if set to < 0, Otsu thresholding is used)
                        watershed_lines=True,                                # Whether watershed should be done (should usually be enabled, but can be disabled if a lot of oversegmentation occurs)
                        min_distance=9,                                      # Minimum distance that will be split by watershed lines (increase if oversegmentation occurs))
                        min_overlap=2,                                       # Minimum overlap between image tiles
                        manage_overlap_mode=2,                               # What to do in overlapping tile regions (0: Use Maximum, 1: Average, 2: Crop)
                        use_gpu=use_gpu_for_inference)                       # Whether to use the GPU during inference


if __name__ == '__main__':
    print(f'Start: {datetime.now()}')

    # Step 0: Configuration and setup
    mp.set_start_method('spawn')
    p0 = mp.Process(target=start_step_0)
    p0.start()
    p0.join()

    # Step 1: Configure and train WGAN
    p1 = mp.Process(target=start_step_1)
    p1.start()
    p1.join()

    # Step 2: Simulate masks
    p2 = mp.Process(target=start_step_2)
    p2.start()
    p2.join()

    # Step 3: Train cycleGAN
    p3 = mp.Process(target=start_step_3)
    p3.start()
    p3.join()

    # Step 4: Simulate fake images and segment real images with cycleGAN
    p4 = mp.Process(target=start_step_4)
    p4.start()
    p4.join()

    # Step 5: Postprocess CycleGAN Output images
    p5 = mp.Process(target=start_step_5)
    p5.start()
    p5.join()

    # Step 6: Train UNet and segment real images
    p6a = mp.Process(target=start_step_6a)
    p6a.start()
    p6a.join()

    p6b = mp.Process(target=start_step_6b)
    p6b.start()
    p6b.join()

    print(f'Finished: {datetime.now()}')
