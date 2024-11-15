import numpy as np
from CoolProp.CoolProp import PropsSI

def get_tracks(Npart: int, Ntime: int, Ndim: int, mean_r: float, T: float, drift: tuple, dt: float, boxsize: tuple,
               gaussian_radius_std: float, ranseed: int = None) -> np.ndarray:
    """
    Function setting up the Brownian motion parameters and returning the tracks of the particles.

    :param Npart: number of particles
    :param Ntime: number of timesteps
    :param Ndim: number of dimensions
    :param r: mean radius of the particles
    :param T: temperature
    :param drift: global drift direction for the particles
    :param dt: time step
    :param boxsize: size of the simulation box in each dimension (in microns)
    :param gaussian_radius_std: standard deviation for the gaussian radius
    :param ranseed: random seed for the random number generator

    :rtype: np.ndarray
    """
    # Set up pseudo-random number generator
    # We will use the 'Mersenne Twister' MT19937
    PRNG = np.random.Generator(np.random.MT19937(seed=ranseed))

    # Reserve the storage for all particle coordinates over all time steps
    ptrack = np.zeros((Npart, Ntime, Ndim))

    # Set the initial particle coordinates by filling the first time index (0)
    # In this case, we consider particles uniformly distributed through the volume
    ptrack[:, 0, :] = (PRNG.random((Npart, Ndim)) - .5) * boxsize

    # Generate all random numbers for the simulation of Brownian motion
    # These are generated all at once, and stored in an array.
    # It may be wise to explicitly specify the order in which the random
    # numbers are drawn.
    norm_noise = PRNG.normal(loc=0.0, scale=1.0, size=(Npart, Ntime - 1, Ndim))

    # Assign a radius to every particle
    if gaussian_radius_std is not None and gaussian_radius_std > 0:
        r = np.random.normal(mean_r, gaussian_radius_std, size=(Npart, 1, 1))
        r[r < 0] = mean_r
    else:
        r = np.ones((Npart, 1, 1)) * mean_r

    # Calculate the diffusion coefficient from the radius and temperature
    eta = PropsSI('V', 'T', T, 'P', 101325., 'water')
    k_B = 1.38065e-23
    D = k_B * T / (6 * np.pi * eta * r)  # Diffusion coefficient
    D *= 1e12  # conversion m²/s => µm²/s

    # Calculate the brownian motion between steps
    Npart, Ntime, Ndims = ptrack.shape
    dx = np.sqrt(2 * D * dt) * norm_noise + np.array(drift) * dt

    for idim in range(Ndims):
        _ = dx[:, :, idim] > boxsize[idim]  # may happen in extremely rare cases?
        dx[_, idim] = 0

    # First value in time is initial position then it's the displacements dx
    ptrack[:, 1:, :] = dx

    # Sum along time from the beginning to the specific index for every index
    ptrack = np.cumsum(ptrack, axis=1)

    # If the brownian path goes outside the box, we teleport de particle back on the other side of the box
    # (can happen multiple times if the box is small)
    for idim in range(Ndims):
        while (out_of_box_p := ptrack[:, :, idim] >= boxsize[idim] / 2).any():
            ptrack[out_of_box_p, idim] -= boxsize[idim]

        while (out_of_box_n := ptrack[:, :, idim] <= -boxsize[idim] / 2).any():
            ptrack[out_of_box_n, idim] += boxsize[idim]

    print("Tracks created")
    return np.dstack((ptrack, np.repeat(r, Ntime, axis=1)))
