from brownian_motion import get_tracks
import numpy as np
import os
from system import Camera, Microscope
import tifffile
from PIL import Image
from particles_PSF import parallelized_image_creation
from ddm_analysis import analyze_ddm
from datetime import datetime

ranseed = None  # integer seed for the pseudo-random number gen or None


# Function normalising every image between 0 and 1
def normalize_stack(stack, without_lower=False):
    # Without lower means we don't set the minimal value to 0 before normalizing
    if without_lower:
        normalized = stack / stack.max()
    else:
        normalized = (stack - stack.min()) / (stack.max() - stack.min())
    return normalized


def noise_stack(stack, SNR):
    # add noise using the signal-to-noise ratio (SNR)
    # np.random.poisson(1, images.shape) has a mean of 1
    signal = np.mean(stack)
    stack += np.random.poisson(1, stack.shape) * signal / SNR
    return stack


# Create a tiff stack from multiple images in a directory
# The function can add_noise
def create_stack_from_tiff(images_folder="images_camera",
                           output_folder="videos", output_filename="out", add_noise=False, save_final_images=False,
                           use_camera_precision=False, saturate=False, *args, **kwargs) -> np.ndarray:
    """
    Function for creating the final video output of the simulation as a TIFF stack.

    :param images_folder: name of folder where images are stored
    :param output_folder: name of folder where video will be stored
    :param output_filename: name of the video output file
    :param add_noise: whether to add noise or not (using SNR value)
    :param save_final_images: whether to save final images after or not after processing
    :param use_camera_precision: whether to encode the images on the same number of bits as the camera
    :param saturate: whether to saturate the images
    :param args: arguments passed to tifffile.imwrite to create the final video
    :param kwargs: keyword arguments passed to tifffile.imwrite to create the final video
    :return: video as multiple images in a numpy array
    """
    images = []  # store images for the video
    paths = []  # store path to each image to replace them after normalising
    for image_file in os.listdir(images_folder):
        if image_file.endswith(".tif") or image_file.endswith(".tiff"):
            # print(f"{images_folder}{image_path}")
            image = tifffile.imread(f"{images_folder}/{image_file}")  # read the image
            images.append(image)  # store it
            paths.append(f"{images_folder}/{image_file}")  # store path
    images = np.array(images)

    if add_noise:
        images = noise_stack(images, SNR)

    # Add saturation to enhance contrast
    if saturate:
        # The 20 * std is arbitrary and purely based on observations
        images[images > (np.mean(images) + 20 * np.std(images))] = np.mean(images) + 20 * np.std(images)

    # Normalize without setting the minimum value to 0
    images = normalize_stack(images, without_lower=True)

    # Encode the image with the same number of bits as the camera
    if use_camera_precision:
        images *= 2 ** camera.encoding_precision
        images = np.floor(images)

    # Replace the old images with the normalised ones
    if save_final_images:
        for path, image in zip(paths, images):
            Image.fromarray(image).save(path)

    # If file doesn't already exist
    if not os.path.isfile(f"{output_folder}/{output_filename}.tiff"):
        # Save the video
        tifffile.imwrite(f"{output_folder}/{output_filename}.tiff", images, *args, **kwargs)
        print(f"Created {output_folder}/{output_filename}.tiff")
    else:
        # If it exists, find an unused filename for the video
        i = 0
        while os.path.isfile(f"{output_folder}/{output_filename}_{i}.tiff"):
            i += 1

        # Save the video
        tifffile.imwrite(f"{output_folder}/{output_filename}_{i}.tiff", images, *args, **kwargs)
        print(f"Created : {output_folder}/{output_filename}_{i}.tiff")

    return images

def launch_movie_generation(params):
    np.random.seed(ranseed)  # Set a seed to create the same simulation every time
    if not os.path.exists("images_camera"):
        os.mkdir("images_camera")
    if not os.path.exists("videos"):
        os.mkdir("videos")

    # Create variables from params dictionary
    Ndim = int(params["Ndim"])
    Npart = int(params["Npart"])
    Ntime = int(params["Ntime"])
    fps = int(params["fps"])

    dt = 1 / fps  # [s] simulation time step
    boxsize = (
    params["boxsize_x"], params["boxsize_y"], params["boxsize_z"])  # [µm, µm, µm] dimensions of simulation box

    r = params["r"]  # [m] radius
    T = params["T"]  # temperature in Kelvin
    gaussian_radius_std = params["gaussian_radius_std"]  # standard deviation for radius [m]

    drift = (params["drift_x"], params["drift_y"], params["drift_z"])  # drift for all the particles [µm/s]

    lightsheet = bool(int(params["lightsheet"]))  # Activation of the lightsheet for the video
    oversampling = int(
        params["oversampling"])  # Defines the upsampling ratio on the image space grid for computations
    SNR = params["SNR"]  # signal-to-noise ratio
    magnification = params["magnification"]

    # Create Camera object
    camera = Camera(
        size=int(params["camera_size"]),
        axial_res=params["axial_res"],
        lateral_res=params["lateral_res"],
        magnification=magnification,
        offset_x=params["offset_x"],
        offset_y=params["offset_y"],
        encoding_precision=int(params["encoding_precision"])
    )

    # Create Microscope object
    microscope = Microscope(
        wavelength=params["wavelength"],
        M=magnification,
        NA=params["NA"],
        ng0=params["ng0"],
        ng=params["ng"],
        ni0=params["ni0"],
        ni=params["ni"],
        ns=params["ns"],
        ti0=params["ti0"],
        tg0=params["tg0"],
        tg=params["tg"],
        zd0=params["zd0"],
        covertop_z=params["covertop_z"]
    )

    # Create tracks
    tracks = get_tracks(Npart, Ntime, Ndim, r, T, drift, dt, boxsize, gaussian_radius_std, ranseed)

    # z must be greater than 0
    tracks[:, :, 2] += boxsize[2] / 2  # Les particules doivent avoir un z > 0

    # Create all images
    parallelized_image_creation(tracks, camera, microscope, boxsize, oversampling, lightsheet=lightsheet)

    # Create filename from date and parameters
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d_%H-%M")
    output_filename = f"video_{date_str}_{lightsheet}_{Npart}_{Ntime}_{r}_{gaussian_radius_std}_{boxsize[0]}-{boxsize[1]}-{boxsize[2]}_{fps}"

    # Create video
    images = create_stack_from_tiff(images_folder="images_camera", output_folder="videos",
                                    output_filename=output_filename,
                                    add_noise=True, save_final_images=False,
                                    use_camera_precision=True, saturate=True,
                                    imagej=True, metadata={'fps': fps, 'finterval': 1 / fps, 'axes': 'TYX'})



if __name__ == '__main__':
    # Simulation parameters
    Ndim = 3  # number of dimensions
    Npart = 1000  # number of particles
    Ntime = 1000  # number of time steps
    fps = 60  # frame per seconds

    dt = 1 / fps  # [s] simulation time step
    boxsize = (130., 130., 40.)  # [µm, µm, µm] dimensions of simulation box

    r = 30e-9  # [m] radius
    T = 293  # temperature in Kelvin

    # Generate radius from a gaussian centered at r with this standard deviation in meter
    # None or 0 for fixed radius
    gaussian_radius_std = 5e-9

    drift = (0, 0, 0)  # drift for all the particles (µm/s)

    lightsheet = True  # Activation of the lightsheet for the video
    oversampling = 1  # Defines the upsampling ratio on the image space grid for computations
    SNR = 5  # signal-to-noise ratio
    magnification = 20

    camera = Camera(
        size=512,
        axial_res=1,
        lateral_res=4.8,
        magnification=magnification,
        offset_x=800,
        offset_y=0,
        encoding_precision=12
    )

    microscope = Microscope(
        wavelength=0.532,  # microns
        M=magnification,
        NA=0.75,  # numerical aperture
        ng0=1.5,  # coverslip RI design value
        ng=1.5,  # coverslip RI experimental value
        ni0=1,  # immersion medium RI design value
        ni=1,  # immersion medium RI experimental value
        ns=1.33,  # specimen refractive index (RI)
        ti0=150,  # microns, working distance (immersion medium thickness) design value
        tg0=170,  # microns, coverslip thickness design value
        tg=170,  # microns, coverslip thickness experimental value
        zd0=180.0 * 1.0e+3,  # microscope tube length (in microns).
        covertop_z=-10  # [µm] focus just 'above' interface (inside the sample medium)
    )
    np.random.seed(ranseed)  # Set a seed to create the same simulation every time
    if not os.path.exists("images_camera"):
        os.mkdir("images_camera")
    if not os.path.exists("videos"):
        os.mkdir("videos")

    # Create tracks
    tracks = get_tracks(Npart, Ntime, Ndim, r, T, drift, dt, boxsize, gaussian_radius_std, ranseed)

    # z must be greater than 0
    tracks[:, :, 2] += boxsize[2] / 2  # Les particules doivent avoir un z > 0

    # Create all images
    parallelized_image_creation(tracks, camera, microscope, boxsize, oversampling, lightsheet=lightsheet)

    # Create filename from date and parameters
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d_%H-%M")
    output_filename = f"video_{date_str}_{lightsheet}_{Npart}_{Ntime}_{r}_{gaussian_radius_std}_{boxsize[0]}-{boxsize[1]}-{boxsize[2]}_{fps}"

    # Create video
    images = create_stack_from_tiff(images_folder="images_camera", output_folder="videos",
                                    output_filename=output_filename,
                                    add_noise=True, save_final_images=False,
                                    use_camera_precision=True, saturate=True,
                                    imagej=True, metadata={'fps': fps, 'finterval': 1 / fps, 'axes': 'TYX'})

    """ Only works if ddm-toolkit is in the project directory """
    # analyze_ddm(images, int(Ntime * 0.8), camera, dt, D_guess=7, model_id=6, print_report=True, show_graphs=True, save_report=True, lightsheet=lightsheet, radius=r, fps=fps)

