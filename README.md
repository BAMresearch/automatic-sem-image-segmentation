Automatic SEM Image Segmentation
================================
The files and images in this repository accompany the following publication:

[B. Ruehle, J. F. Krumrey, V.-D. Hodoroaba, Scientific Reports, _Workflow towards Automated Segmentation of Agglomerated, Non-Spherical Particles from Electron Microscopy Images using Artificial Neural Networks_, **2020**, DOI: 10.1038/s41598-021-84287-6](https://doi.org/10.1038/s41598-021-84287-6)

If you use the files, data, or code in your own work, please cite the above article.

![Image](./ToC.jpg "ToC")

Folder Structure:
-----------------
  * **Archive:** Contains the files discussed and used in the original [publication](https://doi.org/10.1038/s41598-021-84287-6).
    - **Automatic_SEM_Image_Segmentation:** Contains the version of the Python code that was used in the original publication.
	- **Other Scripts:** Contains scripts for calculating the various metrics discussed in the publication.
	- **Trained Neural Network Models:** Contains the fully trained neural networks for SEM image segmentation trained on simulated and manually annotated masks as well as the fully trained classification network trained on manually annotated classes for the masks, as discussed in the publication.
  * **Datasets:** Contains the datasets used in the publication. The dataset is also available via [Zenodo](https://zenodo.org/record/4563942) (DOI: 10.5281/zenodo.4563942). It has the following subfolders:
    - **Electron Microscopy Image Masks:** Contains the manually annotated segmentation and classification masks for the SEM images.
    - **Electron Microscopy Images:** Contains the raw, unprocessed SEM and TSEM images used in the publication
  * **ImageJ Plugin:** Contains an ImageJ Plugin (beta version) that allows to use the fully trained neural networks in inference mode as well as some basic filters directly from ImageJ
  * **Other Scripts:** Contains example files (python and jupyter notebook) demonstrating how to use the trained models for segmenting and classifying SEM images
  * **Releases:** Contains the different versions of the implementation of the workflow.

How to use:
-----------
  1. To train your own network for fully automated segmentation, download the python scripts of the desired version from the `Releases` folder and place them in a new directory.
  2. Create the following sub directories:
     * `Input_Images`
     * `Input_Masks`
  3. Place the SEM images in the `Input_Images` folder and some exemplary particle outlines (white on a black background, see `Archive/Automatic_SEM_Image_Segmentation/Input_Masks` for examples) in the `Input_Masks` directory.
  4. Install the required packages for the version you downloaded by running
	 ```
	 pip install -r requirements.txt
	 ```
	 The requirements are different for the different releases of the workflow. Please also make sure that your python version, as well as the CUDA and cuDNN versions are compatible with the packages from requirements.txt (especially the tensorflow version)
  5. Open the StartProcess.py file and choose the options and parameters you want for training (or skip this step and use the standard options). You probably want to at least set the tile size, and the cycleGAN and UNet batch sizes, though. The tile size should be large enough to fit roughly 100-150 particles (if there are significantly more or less, you might also have to adjust the minimum and maximum number of particles in the simulated masks), and the batch sizes can then usually just be increased until you run out of GPU memory).
  6. Start the process by running `python StartProcess.py`
  
Releases:
-----------------
  * **1.1.0**
    First major new release with various changes. For requirements, see requirements.txt in the subfolder. Tested on Windows 10, python 3.10.6, CUDA 11.2, cuDNN 8.1.0.
    - The code has been completely refactored and streamlined towards the most used applications.
	- The code now runs with much more recent versions of python and the dependencies.
	- The cycleGAN implementation is now based on code published under the Apache license, so all code components use very permissive licenses now.
	- In general, options from the previous version (e.g., from cycleGAN) were implemented as well in the new version. Some default parameters have changed to parameters that give an equal or better performance in our experience, but they can of course be further tuned or changed back if desired.
	- While still using image tiles for training on a GPU (if available), images are now no longer tiled for inference by default (this is possible due to the fully convolutional nature of UNets). Since full images are typically too large to fit on the GPU, inference is now done on a CPU by default. This is usually slower, but the benefits of avoiding tiling/stitching artifacts probably outweigh the performance constraints. This behavior can be changed in the `StartProcess.py` file.
	- Some redundancies have been removed, e.g., having the exact same image files stored at different locations for training and inference in the different steps.
	- The new version offers some new features and several "Quality of Life" changes, such as:
		* possibility to use dataloaders for loading the images dynamically from the hard drive: useful for very large training sets when CPU memory (not GPU memory) becomes limiting.
		* possibility to add Gaussian noise to discriminator layers in cycleGAN to avoid "overtraining" the discriminator and mode collapse in the generator (which is the default setting now), possibility to add a skip connection between input and output layer in the generators (as done in UNets - here it is conceptually similar to using identity mappings), as well as using an "asymmetric" network with binary crossentropy for the generator producing fake masks and mean absolute error for the generator producing fake microscopy images.
		* new options and improved performance for simulating fake masks after WGAN training.
		* making it easier to work with arbitrarily sized images and masks (will be padded automatically to meet shape requirements imposed by the neural network architecture).
		* being more flexible with the image inputs: the code will try to read different image files (not only tif as is in version 1.0.x), will automatically infere the image shape and reshape/convert if necessary (e.g., the masks no longer have to be 8 bit single channel black and white images with black pixels having a value of 0 and white pixels having a value of 255 - they still have to be black and white, but could also be stored in rgb format or have black pixels as 0 and white pixels as 1, etc.).
		* the readability of the code is improved: the main options are given at the beginning, more advanced options can be found deeper within the code, and the naming and structure of the variables and directories used in the different steps is more consistent now.
		* several other small changes
  * **1.0.1**
    Various small updates, bugfixes, and consistency changes. For requirements, see requirements.txt in the subfolder. Tested on Windows 10, python 3.7.4, CUDA 10.1, cuDNN 7.6.
  * **1.0.0**
    First release of the code, together with the original publication (same version as in the `Archive/Automatic_SEM_Image_Segmentation` folder). For requirements, see requirements.txt in the subfolder. Tested on Windows 10, python 3.7.4, CUDA 10.1, cuDNN 7.6.

Examples:
---------
The following images illustrate a few examples of particles consisting of different materials (Au, TiO2, BaSO4, SiO2, FeOx) and imaged with different techniques (SEM, eSEM, TEM), and their segmentation masks that were obtained automatically using the above algorithm (or slight variations thereof).
![Image](./Examples.gif "Examples")

Licenses:
---------
The files of this project are provided under different licenses, please refer also to the [license file](./LICENSE) in the root directory for details. In short, the following licenses are used:  
  * The scientific publication itself is published under the Creative Commons Attribution 4.0 International license [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/legalcode), and was first published in [Scientific Reports](https://doi.org/10.1038/s41598-021-84287-6).  
  * The training and validation data and the derived images and masks are published under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International license ([CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode)).  
  * The ImageJ Plugin is published under the [MIT license](https://opensource.org/licenses/MIT).  

Parts of the code used in this project are based on code published under the following licenses:
  1. Versions 1.0.x
     * The Python implementation of the workflow is published under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0-standalone.html)
     * The WGAN implementation used in these versions are based on the work by A. K. Nain published on the [Keras website](https://keras.io/examples/generative/wgan_gp/) under the [Apache License v2.0](https://www.apache.org/licenses/LICENSE-2.0.txt).  
     * The cycleGAN implementation used in these versions are based on the work by S. T. Karlson published on [GitHub](https://github.com/simontomaskarlsson/CycleGAN-Keras) under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0-standalone.html).  
     * The MultiRes UNet implementation used in these versions are based on the work by N. Ibtehaz published on [GitHub](https://github.com/nibtehaz/MultiResUNet) under the [MIT license](https://opensource.org/licenses/MIT).  
  2. Versions 1.1.x
     * The Python implementation of the workflow is published under the [MIT license](https://opensource.org/licenses/MIT).
     * The WGAN implementation used in these versions are based on the work by A. K. Nain published on the [Keras website](https://keras.io/examples/generative/wgan_gp/) under the [Apache License v2.0](https://www.apache.org/licenses/LICENSE-2.0.txt).  
     * The cycleGAN implementation used in these versions are based on the work by A. K. Nain published on [Keras website](https://keras.io/examples/generative/cyclegan/) under the [Apache License v2.0](https://www.apache.org/licenses/LICENSE-2.0.txt).  
     * The MultiRes UNet implementation used in these versions are based on the work by N. Ibtehaz published on [GitHub](https://github.com/nibtehaz/MultiResUNet) under the [MIT license](https://opensource.org/licenses/MIT).  
