"""
Marius Bousseau 2024
"""


import multiprocessing
import numpy as np
import microscPSF as mpsf
from PIL import Image
from functools import partial, lru_cache
import os
from system import Camera, Microscope
from tqdm import tqdm
from time import perf_counter

# Paramètres du faisceau
w_0_xy = 3e-6  # waist en mètres dans le plan xy
w_z = 10e-6  # waist en mètres dans le plan yz
lambda_ = 532e-9  # Exemple de longueur d'onde du faisceau en mètres
x_0 = 0  # Position du waist en y
x_foc = 700e-6  # Position de la focalisation en y

# Distance de Rayleigh pour les deux plans
x_R_xy = np.pi * w_0_xy ** 2 / lambda_
x_R_z = np.pi * w_z ** 2 / lambda_


# Intensity function (x, y, z in meter)
def I(x: float, y: float, z: float, I_0: float = 1) -> float:
    """
    Function to calculate the light sheet intensity

    :param x: x coordinate (in meter)
    :type x: float
    :param y: y coordinate (in meter)
    :type y: float
    :param z: z coordinate (in meter)
    :type z: float
    :param I_0: Intensity of the laser
    :type I_0: float
    :return: The intensity of the laser at the (x, y, z) position
    :rtype: float
    """
    w_x_xy = w_0_xy * np.sqrt(1 + ((x - x_0) / x_R_xy) ** 2)
    w_x_z = w_z * np.sqrt(1 + ((x - x_foc) / x_R_z) ** 2)

    intensity_xy = I_0 * (w_0_xy / w_x_xy) ** 2 * np.exp(-2 * y ** 2 / w_x_xy ** 2)

    intensity_z = I_0 * (w_z / w_x_z) ** 2 * np.exp(-2 * z ** 2 / w_x_z ** 2)

    return intensity_xy * intensity_z


def create_simulation_image_parallelized(*args: tuple[tuple[np.ndarray, int]], camera: Camera, boxsize: tuple, PSF_radius: float,
                                         oversampling: int, Npart: int, psf_xy_z: np.ndarray, lightsheet: bool) -> None:
    """
    Creates a single simulation image with multiple particles by a linear interpolation of the closest precalculated PSF.

    :param args: arguments specific to each threads in the multiprocessing pool namely the positions of each particle and the timestep.
    :type args: tuple[tuple[np.ndarray, int]]
    :param camera: The camera to use.
    :type camera: Camera object
    :param boxsize: Simulation box size in pixels (x,y,z) (in microns)
    :type boxsize: tuple
    :param PSF_radius: radius in which we calculate the PSF
    :type PSF_radius: float
    :param oversampling: Oversampling factor
    :type oversampling: int
    :param Npart: Number of particles
    :type Npart: int
    :param psf_xy_z: array of precalculated psf in the z-direction
    :type psf_xy_z: np.ndarray
    :param lightsheet: Whether to activate the lightsheet or not
    :type lightsheet: bool
    """
    # Retrieve information from each threads
    ppos = args[0][0]
    timestep = args[0][1]

    # Calculate a padding to handle PSF on the edges of the image
    padding = np.ceil(camera.size_to_pixel(2 * PSF_radius)).astype(int) * oversampling
    if padding % 2 == 0:
        padding += 1

    camera_size = int(camera.size)

    # Creation of the global padded image
    pad_oversampled_image = np.zeros((camera_size * oversampling + 2 * padding, camera_size * oversampling + 2 * padding))
    for i in range(Npart):
        # Retrieve positions of the particle
        part_x = ppos[i, 0]
        part_y = ppos[i, 1]
        part_z = ppos[i, 2]
        part_r = ppos[i, 3]

        # if particle is outside camera, don't take it into account
        camera_real_size = camera.pixel_to_size * camera.size
        if (part_x + PSF_radius < -camera_real_size / 2 or -PSF_radius + part_x > camera_real_size / 2 or
            part_y + PSF_radius < -camera_real_size / 2 or -PSF_radius + part_y > camera_real_size / 2):
            continue

        # Calculate the indices between which part_z lands
        z_index = int(part_z // (camera.axial_res / camera.M))
        z_index_2 = z_index + 1

        # Calculate the weight of each PSF for a linear interpolation
        z_weight = 1 - part_z % (camera.axial_res / camera.M)
        z_weight_2 = 1 - z_weight

        # Combine both PSF with their respective weights
        final_psf = psf_xy_z[z_index] * z_weight + psf_xy_z[z_index_2] * z_weight_2

        # Bigger particles diffract more light
        final_psf *= (2 * part_r * 1E9) ** 6

        if lightsheet:
            # Change the intensity of the PSF based on the intensity function
            # We center the lightsheet at the center of boxsize along Z and not at 0
            final_psf *= I((part_x + camera.offset_x) * 1E-6, (part_y + camera.offset_y) * 1E-6, (part_z - boxsize[2] / 2) * 1E-6)

        ## This part enables sub-pixels precision with linear interpolations but seem to have a great impact on performances especially when the oversampling get bigger
        # Calculate the closest pixel to the particle position
        real_center = np.array([(camera.size_to_pixel(part_y) + camera_size / 2) * oversampling, (camera.size_to_pixel(part_x) + camera_size / 2) * oversampling])
        center = np.round(real_center).astype(int)

        # Calculate the relative index of the closest pixels
        coeffs = real_center - center
        indices_change = np.ceil(coeffs) * 2 - 1

        # Calculate linear interpolation of PSF considering the real position of the particle
        # and add PSF to global image
        for indice_change in [(0, 0), (indices_change[0], 0), (0, indices_change[1]), (indices_change[0], indices_change[1])]:
            coeff = 1
            if indice_change[0] == 0:
                coeff *= 1 - abs(coeffs[0])
            else:
                coeff *= abs(coeffs[0])
            if indice_change[1] == 0:
                coeff *= 1 - abs(coeffs[1])
            else:
                coeff *= abs(coeffs[1])
            top = int(center[0] + padding // 2 + indice_change[0])
            left = int(center[1] + padding // 2 + indice_change[1])
            pad_oversampled_image[top:top + padding, left:left+padding] += final_psf * coeff

    # Remove padding
    camera_image = pad_oversampled_image[padding:-padding, padding:-padding]

    if oversampling > 1:
        nb_rows, nb_cols = camera_image.shape

        # Cut oversampled image into blocks of oversampling*oversampling,
        # shape is the new array shape
        # strides is the number of bytes between 2 adjacent value in each dimension
        # here we use float64 hence the 8 multiplying each values
        camera_image = np.lib.stride_tricks.as_strided(camera_image.copy(),
                                                       shape=(camera_size, camera_size, oversampling, oversampling),
                                                       strides=(nb_cols * oversampling * 8, oversampling * 8, nb_cols * 8, 8)
                                                       )

        # Calculate the mean for each blocks of size oversampling*oversampling
        camera_image = np.mean(np.mean(camera_image, axis=3), axis=2)

    # Save the image
    Image.fromarray(camera_image).save('./images_camera/Combined_PSF_Time_{:05d}.tiff'.format(timestep), format='TIFF')


# Calculates every PSF for different values of z
@lru_cache(maxsize=1)
def create_z_psf(camera: Camera, microscope: Microscope, boxsize: tuple, PSF_radius: float, oversampling: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates the PSF for specific values of z based on camera axial resolution and magnification

    :param camera: The camera to use
    :type camera: Camera object
    :param microscope: The microscope to use
    :type microscope: Microscope object
    :param PSF_radius: radius in which we calculate the PSF
    :type PSF_radius: float
    :param boxsize: Simulation box size in pixels (x,y,z) (in microns)
    :type boxsize: tuple
    :param oversampling: Oversampling factor
    :type oversampling: int

    :return The different values of z and their respective PSF
    :rtype (np.ndarray, np.ndarray)
    """

    mp = microscope.parameters

    # Every z for which we will calculate the PSF
    zs = np.arange(0, boxsize[2] + camera.axial_res / camera.M, camera.axial_res / camera.M)

    # Size of a pixel in the real space
    dxy = camera.pixel_to_size / oversampling

    # Get the bounding box size in pixels
    bounding_box_limit = np.ceil(camera.size_to_pixel(2 * PSF_radius)).astype(int)

    # Take into account oversampling and make it odd because the program works better if the center is well defined
    o_N_xy = bounding_box_limit * oversampling
    if o_N_xy % 2 == 0:
        o_N_xy += 1

    # List to store the PSFs
    psf_xy_z = []

    with tqdm(total=len(zs), desc="Precalculating PSF ", bar_format='{l_bar}{bar:30}{r_bar}{bar:-30b}') as bar:
        for z in zs:  # Calculate PSF for each z
            part_z_arr = np.array(z)
            psf_xyz = mpsf.gLXYZParticleScan(mp, dxy, o_N_xy, part_z_arr,
                                             normalize=False,
                                             wvl=microscope.wavelength,
                                             zd=None,
                                             zv=microscope.covertop_z)

            psf_xy = psf_xyz[0, :, :]

            psf_xy_z.append(psf_xy)
            bar.update(1)

    print("Z PSF created")
    return zs, np.array(psf_xy_z)


def clear_old_images():
    """
    Clear all tiff images in the 'images_camera/' directory in order for the videos to not contain old images
    """
    for file in os.listdir('./images_camera/'):
        if file.endswith('.tiff') or file.endswith('.tif'):
            os.remove(f'./images_camera/{file}')
    print("Old images cleared")


def parallelized_image_creation(tracks: np.ndarray, camera: Camera, microscope: Microscope, boxsize: tuple, oversampling: int, lightsheet: bool) -> None:
    """
    Function for creating the simulation images using multiprocessing.

    :param tracks: Array containing position of particles over time in microns in this order (particle_id, timestep, position)
    :param camera: The camera to use
    :param microscope: The microscope to use
    :param boxsize: Simulation box size in pixels (x,y,z) (in microns)
    :param oversampling: Oversampling factor
    :param lightsheet: Whether to activate the light sheet or not
    """

    Npart = tracks.shape[0]  # Number of particles
    tracks = np.swapaxes(tracks, 0, 1)  # Set time in first axis
    Ntime = tracks.shape[0]
    timesteps = np.arange(Ntime)

    # Bounding box in which we will calculate the PSF (prevent from calculating it for every point in space)
    # 8µm or even 6µm should be enough IF the light sheet is aligned with the focus of the microscope.
    # If the light sheet is far from focus, another approach should probably be used
    PSF_radius = 10  # µm

    # Calculate the PSF for different values of z
    start = perf_counter()
    zs, psf_xy_z = create_z_psf(camera, microscope, boxsize, PSF_radius, oversampling)
    end = perf_counter() - start

    # Homemade condition to keep user up-to-date. There are probably better ways to do it.
    if end - start < 0.5:
        print("Using cached PSF")

    # Clear old images in the images_camera folder
    clear_old_images()

    # Multithreading using 80% of threads available and progress bar
    with multiprocessing.Pool(processes=round(multiprocessing.cpu_count() * 0.8)) as pool:
        with tqdm(total=len(timesteps), desc="Generating images ",
                  bar_format='{l_bar}{bar:30}{r_bar}{bar:-30b}') as pbar:
            for _ in pool.imap_unordered(
                    partial(create_simulation_image_parallelized, camera=camera, boxsize=boxsize, PSF_radius=PSF_radius,
                            oversampling=oversampling, Npart=Npart, psf_xy_z=psf_xy_z, lightsheet=lightsheet),
                    zip(tracks, timesteps), chunksize=10):
                pbar.update()