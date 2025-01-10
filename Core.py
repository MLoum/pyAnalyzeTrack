"""
TODO :
- Reparer les séparations rouge et vert -> Afficher les ratios qq part dans le GUI
- Reparer le multiprocessor
- Reparer la somme de gaussienne
- Voir si j'ai bien la possibilité de travailler par image et non par track (une image avec la position de chaque particule avec leur numéro de track)
- Tracer les corrélations
- Script pour voir tester la technique de la somme de gaussienne.
- Ajouter le code de Marius
"""



import matplotlib.pyplot as plt
import numpy as np
import xml.etree.cElementTree as et
from lmfit import Model
from scipy.fftpack import dst
#import CoolProp
#from CoolProp.CoolProp import PropsSI

#import pandas as pd
import os

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Value
import ctypes
import tifffile
from scipy.special import gamma, gammaln

from itertools import repeat

from multiprocessing import Process, Array

import shelve
import cv2

# This is a GLOBAL variable that is used to track the progress of multicore-calculation.
# Due to the GIL management and  things, this vairable cannot be in the class ("not pickable").


class Track:
    """
    La classe qui contient une track.
    Les données brutes sonts :
    - Les positions (x,y,z) au cours du temps t
    - La qualité du tracking
    - ... (le rayon etc)

    La classe contient plusieurs méthodes pour analyser les données brutes et obtenir les rayons hydrodynamiques etc...
    """
    def __init__(self):
        self.trackMate_id = None
        self.nSpots = None
        self.quality = None
        self.Filter = None
        self.t, self.x, self.y, self.z = None, None, None, None # In num frame and pixel
        self.red, self.green = None, None   # Mean Intensity in the red and green channel for each spot.
        self.r_trackmate = None # Radius in pixel of the particle detected by Trackmate.
        self.diff_x, self.diff_y = None, None
        self.drift_x, self.drift_y = None, None
        self.Dx_gauss, self.Dy_gauss = None, None
        self.Dx_msd, self.Dy_msd = None, None
        self.msd_x = 0
        self.Dx_cov, self.Dy_cov = None, None
        self.var_Dx_cov, self.var_Dy_cov = None, None
        self.result_gauss_x, self.result_gauss_y = None, None

        self.colors = None  # A list of [r,g,b] value, one for each spot

        self.color = None  # Main color of the track (the most present along the track)

        self.red_mean, self.green_mean = 0, 0

        self.r_gauss, self.r_cov_2 = 0, 0
        self.error_r_gauss, self.error_r_cov_2, self.error_r_msd = 0, 0, 0

        # Display attribute
        self.is_filtered = False
        self.is_highlighted = False

    def calculate_displacement(self, pos, algo="None", step=1):
        """
        Return the list of all displacement Δx, defined as Δx = x[i+1] - x[i], while taking into account an eventual drift (cf algo)

        :param algo: Algorithm for drift compensation (https://tinevez.github.io/msdanalyzer/tutorial/MSDTuto_drift.html) :
        "None" : no correction
        "itself" :  remove the mean of the data from all displacements
        "neighbors" : to be implemented
        :return:
        """
        if algo == "None":
            return np.diff(pos,step)
        elif algo == "itself":
            return np.diff(pos  - np.mean(pos),step)
        elif algo == "neighbors":
            # TODO
            return np.diff(pos), np.diff(pos)

    def calculate_D_from_gauss(self, params):
        """
        Estime la valeur du coefficient de diffusion en fittant l'histogramme des déplacements (i.e. Δx) par une gaussienne.
        Plus précisement, la largeur de la gaussienne permet de remonter au coefficient de diffusion.
        La position du centre de la gaussienne permet de remonter à un eventuel drift.

        Mathématiquement la largeur de la gaussienne  est relié à sa variance Var (Δx)
        Var (Δx) = <Δx²> - (<Δx>)²
        <Δx²> est la MSD est on peut en déduire le coefficient de diffusion via  <Δx²> = 2 D Δt

        (<Δx>)² est le carré de la moyenne, on peut en déduire une estimation, c'est le centre de la gaussienne.
        Var (Δx) provient de l'ajustement de l'histogramme par une gaussienne.
        :return:
        """
        kb_ = 1.38E-23
        algo_drift_compensation = params["algo_drift_compensation"]
        min_nb_spot_for_gaussian_fit = params["min_nb_spot_for_gaussian_fit"]
        #isotroptic_diff = params["isotroptic_diff"]
        isotroptic_diff = True

        def gaussian_delta_fit(pos, algo_drift_compensation="None"):
            """
            :param pos: Tableau contenant les positions au cours du temps. Peut être à deux (ou plus dimensions si on prend en compte à la flois l'axe x et l'axe y
            :param algo_drift_compensation:
            :return:
            """

            def gaussian(x, amp, cen, wid):
                """
                A Gaussian (unnormalized) is defined by:

                \[ \exp\left(-\frac{1}{2} \frac{(x-\mu)^2}{\sigma^2}\right) \]

                where \(\mu\) is the mean and \(\sigma\) is the standard deviation, thus \(\sigma^2\) corresponds
                to the variance. Here, \( \text{wid} \) is equal to \( 2\sigma^2 \), i.e. two times the variance
                :param x:
                :param amp:
                :param cen: µ
                :param wid: 2 σ²
                :return:
                """
                return amp * np.exp(-(x - cen) ** 2 / wid)

            if isinstance(pos, list):
                diff_list = []
                for p in pos:
                    diff_list.append(self.calculate_displacement(p, algo_drift_compensation))
                diff = np.concatenate(diff_list)
            else:
                diff = self.calculate_displacement(pos, algo_drift_compensation)

            # In order to get an histogramm we have to choose a binning method we rely on "sturges" (see https://indico.cern.ch/event/428334/contributions/1047042/attachments/919069/1299634/ChoosingTheRightBinWidth.pdf)
            hist, boundaries = np.histogram(diff, bins="sturges")
            boundaries = boundaries[0:-1]
            # We use lmfit for fitting the histogram of the displacement Δx by a gaussian
            gmodel = Model(gaussian, nan_policy='raise')
            # We calculate rough estimate of the amplitude (max) and the width (std) of the gaussian that we use as initial guess
            # and set limits for the parameters
            max_, min_ = np.max(hist), np.min(hist)
            max_bound, min_bound = np.max(boundaries), np.min(boundaries)
            params = gmodel.make_params(cen=np.mean(diff), amp=max_, wid=np.std(diff))
            #FIXME fix initial guess ?
            params["amp"].min = 0
            params["amp"].max = 2 * max_

            params["wid"].min = 0
            params["cen"].min = min_bound

            params["cen"].max = max_bound
            params["wid"].max = max_bound - min_bound

            try:
                result = gmodel.fit(hist, params, x=boundaries)
            except ValueError:
                return None, None, None, None, None, None

            return diff, hist, boundaries, result, result.params["cen"], result.params["wid"]

        def D_from_fit(center, width, nb_spot, params):
            T = params["T"]
            eta = params["eta"]

            timeUnits = params["timeUnits"]
            space_units_nm = params["space_units_nm"]

            mean_squared_SI = (center * space_units_nm * 1E-9) ** 2
            variance_SI = (width * space_units_nm * space_units_nm * 1E-18) / 2 # /2 because the fit gives the width = 2 * σ² = 2 * Variance
            #FIXME Cela me semble faux de ne pas prendre en compte la moyenne. On l'a peut-etre compenser, mais alors on doit avoir presque zero
            msd = variance_SI  # Or: msd = variance_SI + mean_squared_SI, depending on your needs
            #msd = variance_SI + mean_squared_SI # Give slightly lower values...
            D_gauss = msd / (2 * timeUnits)
            r_gauss = kb_ * T / (6 * np.pi * eta * D_gauss)

            error_D = D_gauss * np.sqrt(2/(nb_spot - 1)) # Sampling error see "Improving the quantification of Brownian motion"
            error_r_gauss = kb_ * T / (6 * np.pi * eta * D_gauss**2) * error_D

            return D_gauss, r_gauss, error_r_gauss

        if self.nSpots < min_nb_spot_for_gaussian_fit:
            self.Dx_gauss = -100
            self.Dy_gauss = -100
            self.r_gauss = -1E6
            return



        diff, hist, boundaries, result, center, width = gaussian_delta_fit(self.x, algo_drift_compensation)

        self.gauss_diffX_hist = hist
        self.gauss_diffX_hist_abs = boundaries
        self.gauss_diffX_result = result
        # Test if the fit converged by looking for instance at the first returned value
        nb_spot_x = len(self.x)
        if diff is not None:
            self.Dx_gauss, rx, error_rx = D_from_fit(center.value, width.value, nb_spot_x, params)
        else:
            # A big negative number
            self.Dx_gauss = -100
            rx = -1E6

        diff, hist, boundaries, result, center, width = gaussian_delta_fit(self.y, algo_drift_compensation)
        self.gauss_diffY_hist = hist
        self.gauss_diffY_hist_abs = boundaries
        self.gauss_diffY_result = result
        nb_spot_y = len(self.y)
        # Test if the fit converged by looking for instance at the first returned value
        if diff is not None:
            self.Dy_gauss, ry, error_ry = D_from_fit(center.value, width.value, nb_spot_y,  params)
        else:
            # A big negative number
            self.Dy_gauss = -100
            ry = -1E6

        # If we assume there is an isotropic diffusion, we can concatenate the list of displacement along x and y
        # And fit it with the same gaussian

        diff, hist, boundaries, result, center, width = gaussian_delta_fit([self.x, self.y], algo_drift_compensation)
        self.gauss_diff_hist = hist
        self.gauss_diff_hist_abs = boundaries
        self.gauss_diff_result = result
        nb_spot = nb_spot_x + nb_spot_y
        self.D_gauss, self.r_gauss, self.r_gauss_err = D_from_fit(center.value, width.value, nb_spot, params)

        #self.r_gauss = (rx + ry)/2


    def calculate_D_from_MSD(self, params):
        kb_ = 1.38E-23
        T = params["T"]
        eta = params["eta"]
        timeUnits = params["timeUnits"]
        space_units_nm = params["space_units_nm"]
        algo_drift_compensation = params["algo_drift_compensation"]
        min_nb_spot_for_covariance = params["min_nb_spot_for_covariance"]

        def caculate_MSD_1D(pos, params):

            #FIXME pourquoi les deux codes ne donnent pas les même resultats. On ne peut pas enlever le drift dès le début ?
            # diff = self.calculate_displacement(pos * space_units_nm * 1E-9, params["algo_drift_compensation"])
            # MSD = np.mean(diff ** 2)

            diff = self.calculate_displacement(pos * space_units_nm * 1E-9, algo="None")
            drift = np.nanmean(diff)
            MSD = np.mean((diff - drift) ** 2)

            return MSD

        if self.nSpots < min_nb_spot_for_covariance:
            self.Dx_msd, self.Dy_msd, self.r_msd = -1, -1, -1
            return


        MSD = caculate_MSD_1D(self.x, params)
        self.Dx_msd = MSD / (2 * timeUnits)
        if self.Dx_msd == 0:
            self.Dx_msd, self.Dy_msd, self.r_msd = -1, -1, -1
            return

        r_msd_x = kb_ * T / (6 * np.pi * eta * self.Dx_msd)
        MSD = caculate_MSD_1D(self.y, params)
        self.Dy_msd = MSD / (2 * timeUnits)
        if self.Dy_msd == 0:
            self.Dx_msd, self.Dy_msd, self.r_msd = -1, -1, -1
            return
        r_msd_y = kb_ * T / (6 * np.pi * eta * self.Dy_msd)

        self.r_msd = (r_msd_x + r_msd_y)/2

    def calculate_D_from_covariance(self, params):
        """
        For each track we compute the deplacement of each particule for each frame. Thanks to that we can deduct the
        diffusion coefficient of the brownian movement which give us the size of the particule. We account for the experimentals errors like
        a convection motion in the solution or the errors created by the acquisition of the frames.

        We separate the X axis and the Y axis for each particle which in theory should have no difference since the particles rotates,
        even if the particule isnt spherical the movement on each axes is blended together in the long term.

        We also use the displacement to compute the lags and the covariance of each track, this indicate if the track is close to what would be expected
        of a perfect brownian motion which is supposed to have no memory between each displacement. Of course for each analysis we compute the error.


        <(Δx_n)²> = 2.D.Δt + 2(σ² − 2.D.R.Δt)
                                   -----------------
                                           ^
                                           |
                                           |- Due to the error of localisation σ and motion blur (R : motion blur coefficient)

        < Δx_n . Δx_(n+1)> = - (σ² − 2.D.R.Δt)  NB : Non zero covariance. Should be 0 for a memoryless perfect Brownian motion.
        We can inverse these equations to find :
                <(Δx_n)²> - 2 . σ²
        D = ---------------------------
                2 . (1 - 2.R) . Δt

        σ² = R <(Δx_n)²> + (2.R - 1) . < Δx_n . Δx_(n+1)>

        with the variance on D equals to :

                       2 ∈² + 4 ∈ + 6     4 (1 + ∈)²
        var(D) = D² [ ---------------- + ------------ ]
                             N                N²

        ∈ = σ² / (D Δt) - 2 R

        :param track:
        :return:
        """

        kb_ = 1.38E-23
        R = params["R"]
        T = params["T"]
        eta = params["eta"]
        timeUnits = params["timeUnits"]
        space_units_nm = params["space_units_nm"]
        min_nb_spot_for_covariance = params["min_nb_spot_for_covariance"]
        algo_drift_compensation = params["algo_drift_compensation"]

        def caculate_cov_1D(pos, params):
            # diff = self.calculate_displacement(pos * space_units_nm * 1E-9, algo=params["algo_drift_compensation"])
            # MSD = np.mean(diff ** 2)

            diff = self.calculate_displacement(pos * space_units_nm * 1E-9, algo="None")

            #FIXME implement other drift compensation algorithm ?
            drift = np.nanmean(diff)
            MSD = np.mean((diff - drift) ** 2)

            # covariance = np.mean((diff[:-1]) * (diff[1:]))
            covariance = np.mean((diff[:-1] - drift) * (diff[1:] - drift))

            sigma_square = R * MSD + (2 * R - 1) * covariance
            D_cov = (MSD - 2 * sigma_square) / ((2 - 4 * R) * timeUnits)
            #FIXME why does it happen ?
            if D_cov <= 0:
                return -1, -1, -1, -1
            r_cov_2 = (kb_ * T) / (6 * np.pi * eta * D_cov)

            epsilon = sigma_square / (D_cov * timeUnits) - 2 * R
            N = self.nSpots
            var_D_cov_first_order = (6 + 4 * epsilon + 2 * epsilon ** 2) / N
            var_D_cov_second_order =  (4 * (1 + epsilon)**2) / (N ** 2)
            var_D_cov = D_cov**2 * ( var_D_cov_first_order + var_D_cov_second_order)
            var_D_cov = np.sqrt(var_D_cov)
            error_r_cov = kb_ * T / (6 * np.pi * eta * D_cov ** 2) * var_D_cov
            var_r_cov = error_r_cov

            return D_cov, r_cov_2, var_D_cov, var_r_cov

        if self.nSpots < min_nb_spot_for_covariance:
            self.Dx_cov, self.Dy_cov, self.r_cov, self.error_r_cov = -1, -1, -1, -1
            return -1, -1, -1, -1




        def calculate_cov_2D(posx,posy, params):

            diff_x = self.calculate_displacement(posx * space_units_nm * 1E-9, algo="None")
            diff_y = self.calculate_displacement(posy * space_units_nm * 1E-9, algo="None")

            # FIXME implement other drift compensation algorithm ?
            driftx = np.nanmean(diff_x)
            drifty = np.nanmean(diff_y)
            MSD = np.mean((diff_x-driftx)**2+(diff_y-drifty)**2)

            # covariance = np.mean((diff[:-1]) * (diff[1:]))
            covariance = np.mean((diff_x[:-1]-driftx) * (diff_x[1:]-driftx)+(diff_y[:-1]-drifty) * (diff_y[1:]-drifty))

            sigma_square = R * MSD + (2 * R - 1) * covariance

            D_cov = (MSD - 2 * sigma_square) / ((4 - 8 * R) * timeUnits)
            # FIXME why does it happen ?
            #if D_cov <= 0:
            #    return -1, -1, -1, -1
            r_cov_2 = (kb_ * T) / (6 * np.pi * eta * D_cov)

            epsilon = sigma_square / (D_cov * timeUnits) - 2 * R
            N = self.nSpots
            var_D_cov_first_order = (6 + 4 * epsilon + 2 * epsilon ** 2) / N
            var_D_cov_second_order = (4 * (1 + epsilon) ** 2) / (N ** 2)
            var_D_cov = D_cov ** 2 * (var_D_cov_first_order + var_D_cov_second_order)
            var_D_cov = np.sqrt(var_D_cov)
            error_r_cov_1 = kb_ * T / (6 * np.pi * eta * (D_cov + var_D_cov))
            error_r_cov_2 = kb_ * T / (6 * np.pi * eta * (D_cov - var_D_cov))
            var_r_cov = np.absolute((error_r_cov_1 - error_r_cov_2) / 2)

            return D_cov, r_cov_2, var_D_cov, var_r_cov

        self.D_cov, self.r_cov, self.var_D_cov, self.var_r_cov = calculate_cov_2D(self.x,self.y, params)
        self.error_r_cov = self.var_r_cov

    def get_full_MSD(self, positions):
        """
        Calculate the Mean Squared Displacement (MSD) from successive positions of a particle.

        Parameters:
        positions (tuple of np.ndarray): A tuple containing two arrays, (x, y) positions of the particle.

        Returns:
        np.ndarray: An array of MSD values for each time interval.
        """
        x, y = positions
        N = len(x)
        self.full_msd = np.zeros(N)

        """
        for t in range(1, N):
            dx = x[t:] - x[:-t]
            dy = y[t:] - y[:-t]
            squared_displacements = dx ** 2 + dy ** 2
            self.full_msd[t] = np.mean(squared_displacements)
        """
        for t in range(1, N):
            deltas = np.vstack((x[t:] - x[:-t], y[t:] - y[:-t]))
            squared_displacements = np.sum(deltas ** 2, axis=0)
            self.full_msd[t] = np.mean(squared_displacements)

        return self.full_msd
    def extract_colored_ROI(self, video_frames, color):

        # NB : Vectorized
        # Keep only the frame corresponding to the track
        if video_frames is None:
            # No movie associated with the track. We keep the default values
            return

        selected_frames = video_frames[self.t]

        #Array of Top left corner coordinates
        x_start = max(self.x - self.r_trackmate, 0)
        y_start = max(self.y - self.r_trackmate, 0)

        #Array of Bottomp right corner coordinates
        x_end = min(self.x + self.r_trackmate, selected_frames.shape[1] - 1)
        y_end = min(self.y + self.r_trackmate, selected_frames.shape[0] - 1)

        if color == "red":
            self.red = np.mean(selected_frames[y_start :y_end+1, x_start:x_end+1])
            self.red_mean = np.mean(self.red)
        elif color == "green":
            self.green = np.mean(selected_frames[y_start :y_end+1, x_start:x_end+1])
            self.green_mean = np.mean(self.green)




    def Tensor_brownian(self,params):
        space_units_nm = params["space_units_nm"]
        diff_x = self.calculate_displacement(self.x * space_units_nm *1E-9, algo="None")
        diff_y = self.calculate_displacement(self.y * space_units_nm  *1e-9, algo="None")
        Tensor = np.zeros((2,2))
        Tensor[0][0] = np.mean(diff_x ** 2) - np.mean(diff_x) ** 2
        Tensor[0][1] = np.mean(diff_x * diff_y) - np.mean(diff_x) * np.mean(diff_y)
        Tensor[1][0] = Tensor[0][1]
        Tensor[1][1] = np.mean(diff_y ** 2) - np.mean(diff_y) ** 2
        eigenvalues, eigenvectors = np.linalg.eig(Tensor)
        R1 = np.sqrt(eigenvalues[0])
        R2 = np.sqrt(eigenvalues[1])
        RG = np.sqrt(R1 + R2)
        self.asym = -np.log10(1 - ((R1 - R2) ** 2 )/ (2 * RG ** 2))
        self.asym = format(self.asym*1e9, ".2e")
        #print("R2",R2)



"""
En dehors des class pour pouvoir le sérialiser lors des processus multicore
"""

def init_track(data, params):
    """
    Using ProcessPoolExecutor has the disadvantage that each child process runs in its own memory space,
    separate from the parent process. This means that any changes made to an object in the child process
    will not be reflected in the parent process.
    Solution:
    Return the modified object: return the modified object from the function you're mapping,
    and then use these returned objects to update the list in the parent process.

    :param data:
    :return:
    """
    track = Track()
    #TODO the value of the columns are yet "magic numbers", we should identify them in the header instead of relying on a given order
    #FIXME est ce que 1 est le bon index pour le trackID ?
    track.trackMate_id = data[0, 1] # Here we take only one value, since its should be all the same.
    track.spot_ids = data[:, 0]
    
    
    track.quality = data[:, 2]
    track.x = data[:, 3]
    track.y = data[:, 4]
    track.z = data[:, 5]
    track.t = data[:, 6]
    track.f = data[:,7]
    track.r_trackmate = data[:, 8] # This is the "RADIUS" column, the value is fixed and correspond to the one given by the user during the spot detection.
    # track.r_trackmate = data[:, 17] # This is ESTIMATED_DIAMETER, should be better.
    #TODO Add contrast and/or SNR.

    track.nSpots = track.x.size

    track.calculate_D_from_gauss(params)
    track.calculate_D_from_MSD(params)
    track.calculate_D_from_covariance(params)
    track.Tensor_brownian(params)


    # Ini value for colors -> no color since the tracking was done with a grey value movie
    track.colors = [0, 0 ,0]
    track.color = None

    #FIXME external counter in order to monitor calculation
    #global task_counter
    #task_counter += 1
    #print("task_counter = ", task_counter)

    return track



class Spot():
    """
    A spot is a detected particle. It belong to a track.
    """

    def __init__(self):
        self.x = None # position in ?
        self.y = None # position in ?
        self.parent_track = None 
        self.num = None
        self.id = None
        self.color = [0, 0, 0] # RGB

class Frame():
    """
    The data can be organized as track where we follow a particle during its tracking
    The same data can also be organized as frames. A frame is a single image of the movie.
    More precisely, it does contain the pixel data of the corresponding image but also the position 
    of all its spot and for each spot the corresponding track
    """
    
    def __init__(self):
        self.img = None # The raw pixel data of the frame in [R,G,B] format
        self.num = -1 # The frame number of the full movie
        self.spots = None # List of the spots detected inside this frame
    
               

class AnalyzeTrackCore():
    """

    """
    def __init__(self):
        self.Filter = 10    # ?
        self.overlap = 0    # ?
        self.overlap_time = []
        self.number_tracks_green = []
        self.number_tracks_red = []

        self.min_nb_spot_for_gaussian_fit = 10  #FIXME Hardcoded
        self.min_nb_spot_for_covariance = 10  # FIXME Hardcoded
        self.tracks = None # List of the track object
        self.nTracks = None
        self.frames = None # List of Frame objects
        self.nFrames = 0
        self.frameInterval = None
        self.space_units_nm = 240 # pour la caméra ximéa -> FIXME Hardcoded
        #self.space_units_nm = 172.5 #pour la camera IDS # in nm
        self.frame_rate = 1 / 120 #   # FIXME Harcoded
        self.T = 293    #in K
        #self.eta = PropsSI('V', 'T', self.T, 'P', 101325., 'water')    # in Pa.s
        self.eta = 0.001
        self.sigma_already_known = False
        self.R = 1/6    #FIXME ?
        self.division = 10
        self.tracker = 0

        self.is_computing = False

        self.executor = None

        self.algo_drift = "itself"

        # Tracking of the calculations progresses
        self.tasks_completed = Value(ctypes.c_int, 0)
        self.current_task = "Idle"
        self.task_counter = 0

        self.kb = 1.380649E-23


    def viscosity_from_temperature(self, solvent="water", T=293):
        # TODO tabuler la viscosité de l'eau.
        if solvent == "water":
            return 1E-3

        return None

    def radius_nm_from_coeff_diff(self, D_SI):
        return self.kb * self.T / (6 * np.pi * self.eta * D_SI)*1E9

    def change_filter(self, Valeur):
        self.Filter = int(Valeur.get())


    def load_data(self, params):
        self.is_computing = True

        self.load_TrackData_txt(params)

        if params["filepath_video"] != "None":
            self.load_VideoData(params)
        else:
            self.red_frames = None
            self.green_frames = None

        self.is_computing = False

    def load_TrackData_txt(self, params):
        """
        We use Trackmate to get the tracks. Its format output is a text file with :
        - a header line with name of the paremeter separated by a space :
        e.g. LABEL ID TRACK_ID QUALITY POSITION_X POSITION_Y POSITION_Z POSITION_T FRAME RADIUS VISIBILITY MANUAL_SPOT_COLOR MEAN_INTENSITY_CH1 MEDIAN_INTENSITY_CH1 MIN_INTENSITY_CH1 MAX_INTENSITY_CH1 TOTAL_INTENSITY_CH1 STD_INTENSITY_CH1 CONTRAST_CH1 SNR_CH1

        - When the txt data has been converted we analyze it in order to get its Diffusion Coefficient

        :param params: A dictionnary with a key filepath_track for the path of the txt file exported from Trackmate
        :return: FIXME !
        """
        self.current_task = "Track Data loading"

        filepath = params["filepath_track"]

        def find_track_id_index(filepath):
            """
            Sub funciton of "load_TrackData_txt" that returns the nb of columns and the position of two columns : TRACK_ID and POSITION_T
            :param filepath:
            :return:
            """
            # TODO chercher aussi les headers, x, y, diameter, etc...
            with open(filepath, 'r') as file:
                # Read the first line, which is the header
                header_line = file.readline().strip()

            # Split the line by spaces to get column names
            columns = header_line.split(" ")

            # Count the number of columns
            num_columns = len(columns)

            # Find the index of the "TRACK_ID" column
            try:
                track_id_index = columns.index("TRACK_ID")
            except ValueError:
                print("The 'TRACK_ID' column does not exist in the file.")
                return None, None, None

            # Find the index of the "POSITION_T" column. T stands for time
            try:
                position_T_index = columns.index("POSITION_T")
            except ValueError:
                print("The 'POSITION_T' column does not exist in the file.")
                return None, None, None

            return num_columns, track_id_index, position_T_index

        num_columns, track_id_index, position_T_index = find_track_id_index(filepath)
        if num_columns is None:
            print("Error during the track data loading : no header in the text file")
            return None

        # Load file to numpy array
        # We skip the first line which is the text header (skiprows=1)
        # We don't take in account the first column (LABEL_ID) which contains text and that is redundant with ID.
        # Typical Header :
        # Label    ID       TRACK_ID QUALITY POSITION_X POSITION_Y POSITION_Z POSITION_T FRAME RADIUS VISIBILITY MANUAL_COLOR MEAN_INTENSITY MEDIAN_INTENSITY MIN_INTENSITY MAX_INTENSITY TOTAL_INTENSITY STANDARD_DEVIATION ESTIMATED_DIAMETER CONTRAST SNR
        # ID10975  10975	0	     2.451	 317.287	192.140	0  0	0	5	1	-10921639	27.660	26	8	58	2683	10.804	5.096	0.279	1.116
        raw = np.loadtxt(filepath, skiprows=1, usecols=np.arange(1, num_columns-1))

        # Since we don't take in account Label, we need to update the Track_ID position
        track_id_index -= 1
        position_T_index -= 1

        nb_lines = np.shape(raw)[0]
        # print(nb_lines)

        # Tracks are listed by ascending trackID and then chronologicaly (MOST of the time, in some txt files, the track where misteriously mixed...)

        # We search for the positions in the raw numpy array of each unique track based on their Track_ID
        unique, unique_index = np.unique(raw[:, track_id_index], return_index=True)
        self.nTracks = np.size(unique) - 1  #FIXME why -1 ?

        # Split the raw data into a list of sub-arrays, one per unique track
        data_splits_per_track = [raw[unique_index[i]:unique_index[i + 1], :] for i in range(self.nTracks)]

        # Global variable for multiprocessor management
        global tasks_completed
        tasks_completed = 0

        # Preparing params for multiprocessor calculation of each tracks
        #TODO transfer the nb of the columns
        #FIXME eta(T)
        params = {
            "T": self.T,
            "eta": self.eta,
            "timeUnits": self.frame_rate,   # TODO rename it frame_rate instead of timeUnits which is ambigouys
            "space_units_nm": self.space_units_nm,
            "R": self.R,    # what is R ?
            "algo_drift_compensation": self.algo_drift,
            "min_nb_spot_for_gaussian_fit": self.min_nb_spot_for_gaussian_fit,
            "min_nb_spot_for_covariance": self.min_nb_spot_for_covariance
        }



        # Create a ProcessPoolExecutor and populate the Track objects in parallel
        num_cores = os.cpu_count()

        self.current_task = "Processing tracks"

        #FIXME j'ai cassé le multiproc !
        # with ProcessPoolExecutor(max_workers=int(num_cores*0.75)) as executor:
        #     self.tracks = list(executor.map(init_track, data_splits, repeat(params)))

        self.tracks = list(map(init_track, data_splits_per_track, repeat(params)))

        # Filter tracks with not enough spots for reliable tracking analysis
        self.tracks = [track for track in self.tracks if track.nSpots >= self.min_nb_spot_for_covariance]
        self.nTracks = len(self.tracks)

        self.current_task = "Arranging tracks in frame structure"

        #for data in data_splits:
        #    self.tracks.append(init_track(data, params))

        # Create the Frame structure where data are not organized as tracks but as frame
        # Since the data is already organized by track, we don't go back to raw data that we would organize by time
        # but we capitalize on the already existing track organization

        # Combien y a til de frames ? Autant qu'il y a d'image dans le fichier. Si je n'ai pas le film, je peux me baser sur l'index t max dans les tracks.



        max_t = -1
        for track in self.tracks:
            t_max_track = max(track.t)
            if t_max_track > max_t:
                max_t = t_max_track
        nb_frame = int(max_t)
        self.frames = [Frame() for i in range(nb_frame + 1)]

        for frame in self.frames:
            frame.spots = []
        for track in self.tracks:
            for i in range(track.nSpots):
                spot = Spot()
                spot.parent_track = track
                spot.id = track.spot_ids[i]
                spot.x = track.x[i]
                spot.y = track.y[i]

                num_frame = int(track.t[i])
                if num_frame >= nb_frame + 1:
                    dummy = 1
                self.frames[num_frame].spots.append(spot)
        self.get_stats()


        self.current_task = "idle"

    def get_histogramm_color(self, params,method,type="standard"):
        if type == "standard":
            #method = params["method"]
            radius_list =[]
            for track in self.tracks:
                if not track.is_filtered:
                    if method == "Red":
                        if track.color == 0:
                            radius_list.append(track.r_cov)
                    elif method == "Green":
                        if track.color == 1:
                            radius_list.append(track.r_cov)
                    elif method == "Blue":
                        if track.color == 2:
                            radius_list.append(track.r_cov)
            self.y_histo, self.x_histo = np.histogram(radius_list,bins = 40)#,range=(params["r_min_nm"],params["r_max_nm"]))
            return self.x_histo, self.y_histo


    def get_histogramm(self, params,type="standard", method = "Covariance"):
        if type == "standard":
            #method = params["method"]
            radius_list =[]
            for track in self.tracks:
                if not track.is_filtered:
                    if method == "Covariance":
                        radius_list.append(track.r_cov)
                    elif method == "GaussianFit":
                        radius_list.append(track.r_gauss)
                    elif method == "MSD":
                        radius_list.append(track.r_msd)

            self.y_histo, self.x_histo = np.histogram(radius_list,bins = 40)#,range=(params["r_min_nm"],params["r_max_nm"]))
            return self.x_histo, self.y_histo

        elif type == "gauss":
            # We add a NORMALIZED gaussian for each track. This gaussian is centered
            """
            \frac{1}{\sigma \sqrt{2\,\pi}}\, \mathrm{e}^{-\frac{\left(x-\mu\right)^2}{2\sigma^2}}
            where sigma is the standart deviation wheras error_r_cov is the variance
            """
              #FIXME ? Hardcoded
            r_min_nm = params["r_min_nm"]
            r_max_nm = params["r_max_nm"]
            nb_pt_in_histo = 10*r_max_nm
            self.x_histo = np.linspace(r_min_nm, r_max_nm, nb_pt_in_histo)*1E-9
            self.y_histo = np.zeros(nb_pt_in_histo)
            for track in self.tracks:
                if not track.is_filtered:
                    #FIXME track.error_r_cov <-> Variance
                    #FIXME l'erreur de la covariance est bien trop petite.
                    variance = track.error_r_cov
                    r = track.r_cov
                    gaussian = 1 / (variance * np.sqrt(2 * np.pi )) * np.exp(-(self.x_histo - r) ** 2 / (2 * variance **2))
                    gaussian *= 1E9 # to nm
                    self.y_histo += gaussian
            return self.x_histo, self.y_histo

        elif type == "gauss Red" :
            r_min_nm = params["r_min_nm"]
            r_max_nm = params["r_max_nm"]
            nb_pt_in_histo = 10 * r_max_nm
            self.x_histo = np.linspace(r_min_nm, r_max_nm, nb_pt_in_histo) * 1E-9
            self.y_histo = np.zeros(nb_pt_in_histo)
            for track in self.tracks:
                if not track.is_filtered:
                    if track.color == 0 :
                        variance = track.error_r_cov
                        r = track.r_cov
                        gaussian = 1 / (variance * np.sqrt(2 * np.pi)) * np.exp(-(self.x_histo - r) ** 2 / (2 * variance ** 2))
                        gaussian *= 1E9  # to nm
                        self.y_histo += gaussian
            return self.x_histo, self.y_histo

        elif type == "gauss Green" :
            r_min_nm = params["r_min_nm"]
            r_max_nm = params["r_max_nm"]
            nb_pt_in_histo = 10 * r_max_nm
            self.x_histo = np.linspace(r_min_nm, r_max_nm, nb_pt_in_histo) * 1E-9
            self.y_histo = np.zeros(nb_pt_in_histo)
            for track in self.tracks:
                if not track.is_filtered:
                    if track.color == 1 :
                        variance = track.error_r_cov
                        r = track.r_cov
                        gaussian = 1 / (variance * np.sqrt(2 * np.pi)) * np.exp(-(self.x_histo - r) ** 2 / (2 * variance ** 2))
                        gaussian *= 1E9  # to nm
                        self.y_histo += gaussian
            return self.x_histo, self.y_histo

        elif type == "MLE":
            """
            Based on Improved nano-particle tracking analysis John G WalkerMeas. Sci. Technol. 23 (2012) 065605 doi:10.1088/0957-0233/23/6/065605
            """
            return self.get_histogram_MLE(3,params)

        elif type == "MLE_red":
            return self.get_histogram_MLE(0,params)
        elif type == "MLE_green":
            return self.get_histogram_MLE(1,params)
        elif type == "MLE_blue":
            return self.get_histogram_MLE(2,params)



    def get_histogram_MLE(self, color,params):
        """
        Draw an histogramm of the measured radius of the particle weighting each track with its number of spot.
        More precisely, we use an Minium LogLikelyHood estimation.
        Based on Walker, J.G., 2012. Improved nano-particle tracking analysis. Measurement Science and Technology, 23(6), p.065605.

        :return:
        """
        bin_size_nm = 5 # NB : hardcoded
        delta_R = bin_size_nm * 1E-9 # SI -> m

        # self.T, self.eta, self.timeUnits



        # Avoir les données brutes
        # nanotrackJ retourne le coefficient de diffusiond comme (D1 + D2) /2, mesurer sur les axes x et y.
        # Puis avant d'envoyer vers l'algorithme de Walker, on a :
        # 		double pixelSquared_to_E10x_cmSquared=nmPerPixel*nmPerPixel*Math.pow(10,-4);
        # 		dc = diffCoeffEst.getDiffusionCoefficient(this, driftx, drifty) * pixelSquared_to_E10x_cmSquared;
        # double msd = d * 4.0 / framerate; // Diffusionkoeffizient zurückrechnen

        # Trouver les min et max en MSD et nbSpots.

        MSDs = [] # List of diffusion coefficient (averaged on x and y).
        ks = [] # List of nb of spot in the track
        rs = []
        # conv_fact = 1E-10
        for track in self.tracks:
            if track.is_filtered or track.r_cov < 0:
                continue
            if track.D_cov == -1 :
                continue
            if track.color == color :
                D = track.D_cov   # * conv_fact -> SI m²/s

                MSD = D * 4 / self.frame_rate # SI m²
                MSDs.append(MSD)
                ks.append(track.nSpots)
                r = self.radius_nm_from_coeff_diff(D)
                rs.append(r)
            elif color == 3 or color == None :
                D = track.D_cov  # * conv_fact -> SI m²/s
                print(D)
                MSD = D * 4 / self.frame_rate  # SI m²
                MSDs.append(MSD)
                ks.append(track.nSpots)
                r = self.radius_nm_from_coeff_diff(D)
                rs.append(r)

        k_max = max(ks)
        k_min = min(ks)
        MSD_max = max(MSDs)
        r_max_nm = max(rs)

        # LUT for log calculation
        logMapK = np.full(k_max + 1, np.nan)  # Cache pour log(k)
        logMapGammaK = np.full(k_max + 1, np.nan)  # Cache pour log(gamma(k))



        #r_max_nm_user = params["r_max_nm"]
        r_max_nm_user = 200 #FIXME
        if r_max_nm_user != 0:
            r_max_nm = r_max_nm_user

        bin_num = int(r_max_nm / bin_size_nm)

        #TODO, calculer les LUT dès maintenant et pas au fur et à mesure
        theta_LUT = np.zeros(bin_num)
        log_theta_LUT = np.zeros(bin_num)
        log_k_LUT = np.zeros(k_max + 1)
        logMapGammaK = np.zeros(k_max + 1)

        hist_bin_number = int (np.sqrt(len(MSDs))) # "square root rule" for the number of MSD bins (https://medium.com/@maxmarkovvision/optimal-number-of-bins-for-histograms-3d7c48086fde)
        delta_B = MSD_max / hist_bin_number

        histogram_MSD = np.zeros(hist_bin_number)
        Nk = np.zeros(k_max + 1)

        # Préparartion des histogrammes Nk -> combien il y a de track avec k et histogram_MSD ->
        for i in range(len(MSDs)):
            Nk[ks[i]] += 1
            idx = int(MSDs[i] / delta_B - 0.001) # 0.001 ?
            histogram_MSD[idx] += 1

        def probMSD(msd, k, r):
            """
            On calcul le log pour ensuite prendre l'exponentielle pour des questions de nombre très grands...
            """
            #                           k_n (k_n \Delta_n)^{k_n - 1} exp(-k_n \Delta_n / \theta_r)
            # P_d(\Delta_n; k_n, r) = --------------------------------------------------------------
            #                              \theta_r^{k_n} + \Gamma(k_n)

            # p_msd = (log(k) + (k-1) * (log(k) + log(msd)) - k * msd / theta) - (k * log(theta) + log(gamma(k)))
            # theta_r = 2 K R \deltat / (3 * pi * eat * t + 2 simga_e^2

            # TODO LUT de theta et log theta
            theta = (2 * self.kb * self.T / self.frame_rate) / (3 * np.pi * self.eta * r)

            # pmsd = k * np.power((msd * k), k-1) * np.exp(- k * msd / theta) / (np.power(theta, k) * gamma(k))

            log_pmsd = (np.log(k) + (k - 1) * (np.log(k) + np.log(msd)) - (k * msd / theta)) - (k * np.log(theta) + gammaln(k))
            #TODO lut de msd
            # log_pmsd = (logK(k) + (k - 1) * (logK(k) + np.log(msd)) - (k * msd / theta)) - (k * np.log(theta) + logGammaK(k))

            pmsd_via_log = np.exp(log_pmsd)
            return pmsd_via_log

        def logK(k):
            """
            Using a LUT for log calculation
            """
            if not np.isnan(logMapK[int(k)]):
                return logMapK[int(k)]
            logMapK[int(k)] = np.log(k)
            return logMapK[int(k)]

        def logGammaK(k):
            """
            Using a LUT for log calculation
            """
            if not np.isnan(logMapGammaK[int(k)]):
                return logMapGammaK[int(k)]
            logMapGammaK[int(k)] = gammaln(k)
            return logMapGammaK[int(k)]

        # Le temps de calcul est trop long, et j'ai l'impression que l'on recalcule sans cesse la même chose. On va donc faire une LUT
        # En entrée, le numéro de la bin d'histogramme de la MSD, k le nombre de spot dans la track et le numero de bin de l'histogramme en r
        prob_MSD_LUT = np.zeros((len(MSDs), k_max, bin_num))

        # Non ! Ma LUT est trop débile. ce n'est pas la peine de calculer tous les couples Delta_n, k_n mais seul ceux qui sont présents expéiremntalement
        # La table n'est que 2D, première dimension le numéro de la track n  et ensuite le rayon de la densité de proba.
        nb_track = len(MSDs)
        prob_MSD_LUT = np.zeros((nb_track, bin_num))
        # # TODO paralellize, vectorize ?
        for n in range(nb_track):
            for r_idx in range(bin_num):
                prob_MSD_LUT[n, r_idx] = probMSD(MSDs[n], ks[n], (r_idx + 1) * delta_R)

        #plt.plot(prob_MSD_LUT[20, :])
        #plt.show()

        # Il faut aussi une LUT différente pour lorsque l'on reconstruit l'histogramme des MSD
        prob_hist_MSD_LUT = np.zeros((hist_bin_number, k_max - k_min + 1, bin_num))
        for b in range(hist_bin_number):
            for k in range(k_min, k_max + 1):
                for r_idx in range(bin_num):
                    prob_hist_MSD_LUT[b, k-k_min, r_idx] = probMSD((b + 1) * delta_B, k, (r_idx + 1) * delta_R)

        #print("pob_hist_MSD_LUT")
        #plt.plot(prob_hist_MSD_LUT[:, 50, 20])
        #plt.title("prob_hist_MSD_LUT")
        #plt.show()

        # La densité de probabilité en rayon P_r, ce que l'on cherche est discrétisée en M points indéxé par m, avec r_m = m \Delta r (delta_R ici)

        # LUT for theta avec le rayon r qui prend toujours les mêmes valeurs discretes :  (r_idx + 1) * delta_R
        for r_idx in range(bin_num):
            log_theta_LUT[r_idx] = (2 * self.kb * self.T / self.frame_rate) / (3 * np.pi * self.eta * (r_idx + 1) * delta_R)


        # # TODO paralellize, vectorize ?
        # for msd_idx in range(len(MSDs)):
        #     for k in range(1, k_max): # pas zero car on va calculer le log. En soit, c'est inutile de calculer avant k > 10 ...
        #         for r_idx in range(bin_num):
        #             prob_MSD_LUT[msd_idx, k, r_idx] = probMSD(MSDs[msd_idx], k, (r_idx + 1) * delta_R)


        def getHistogramML(pm):
            """
            TODO vectorize ?
            :param pm:
            :return:
            """
            histMl = np.zeros(hist_bin_number)
            sumpm = np.sum(pm)
            for b in range(hist_bin_number):
                outersum = 0
                for k in range(k_min, k_max + 1):
                    innersum = 0
                    for m in range(len(pm)):
                        # probMSD(msd, k, r)
                        # Σ_{m = 1}^M P_d(\Delta_n ; k_n, r_n) P_m / (Σ_{m = 1}^M P_m)
                        # innersum += (probMSD((b + 1) * delta_B, k, (m + 1) * delta_R) * delta_B * pm[m]) / sumpm
                        innersum += prob_hist_MSD_LUT[b, k-k_min, m] * delta_B * pm[m] / sumpm
                    # 1/N * Σ_[n=1}^N
                    outersum += Nk[k] * innersum
                histMl[b] = outersum
            return histMl

        def getChiSquared(pm):
            sumchi = 0
            histML = getHistogramML(pm)
            # for b in range(hist_bin_number):
            #     diff = histogram_MSD - histML[b]
            #     sumchi += diff**2 / histML[b]
            diff = histogram_MSD - histML
            sumchi = np.sum((diff ** 2) / histML)
            return sumchi

        # initialisation de la densité de proba -> uniforme
        # dens = np.full(hist_bin_number, 1.0 / hist_bin_number)
        dens = np.full(bin_num, 1.0 / bin_num)
        #print("dens",dens)
        lastChiSquared = getChiSquared(dens)
        #("LastChiSquared",lastChiSquared)
        changeChiSquared = np.inf

        # Ici, il faut que j'utilise les MSD des datas Delta et les nombres de spot k
        # Par contre quelle MSD prendre, j'ai celle en x et en y, la moyenne des deux ? Il faut regarder dans le code de nanotrackj
        # Je remonte le code nanotrackJ_.java

        # Je trouve que c'est msd = (Dx + Dy)/2 * nmPerPixel * nmPerPixel * 1E-4 * 4 / Framerate
        # Ce qui est raccord avec une diffusion 2D où MSD = 4 * D * Delta t

        """
        double d = tracks.get(i).getDiffusionCoefficient(doCorrectDrift, useKalman);
         double msd = d*4.0/framerate; //Diffusionkoeffizient zurückrechnen
        data.add(msd);
        data.add((double)tracks.get(i).size());
        WalkerMethodEstimator walker = new WalkerMethodEstimator(dataarray, temp, visk, framerate,maxWalkerHistogrammDiameter);

        Puis dans track.j
            private double getDiffusionCoefficient(double driftx, double drifty){
            double pixelSquared_to_E10x_cmSquared=nmPerPixel*nmPerPixel*Math.pow(10,-4);
            dc = diffCoeffEst.getDiffusionCoefficient(this, driftx, drifty) * pixelSquared_to_E10x_cmSquared;
            return dc;
        }

        puis :

            @Override	
            /**
             * Calculates the Diffusion Coefficient as described in:
             * Vestergaard, C., 2012. 
             * Optimal Estimation of Diffusion Coefficients from Noisy Time-Lapse-Recorded Single-Particle Trajectories. 
             * Technical University of Denmark.
             */
            public double getDiffusionCoefficient(Track track, double driftX,
                    double driftY) {
                this.driftx = driftX;
                this.drifty = driftY;
                this.track = track;

                if(track.size()==1){
                    return 0;
                }
                double[] covData = getCovData(track, 0, driftX, driftY);

                return covData[0];
            }

        double D1 = termXA+termXB;	
        double D2 = termYA+termYB;
        double D = (D1+D2)/2;

        puis de covariance estimator :
                double[] data  = new double[3]; //[0] = Diffusioncoefficient, [1] = LocNoiseX, [2] = LocNoiseY
        data[0] = D;
        data[1] = R*msdX + (2*R-1)+covX;
        data[2] = R*msdY + (2*R-1)+covY;
        return data;
        """


        while changeChiSquared > 0.01:
            for m in range(len(dens)):
                #print("length(dens)",len(dens))
                sumpm = np.sum(dens)
                sum_over_n = 0
                nb_track = len(MSDs)
                for n in range(nb_track):
                    denominator = 0
                    # probMSD(msd, k, r)
                    # numerator = probMSD(MSDs[n], ks[n], (m + 1) * delta_R)   # numérateur
                    numerator = prob_MSD_LUT[n, m]
                    #print("prob_MSD_LUT[n, m]", numerator)
                    # numérateur

                    # dénominateur
                    for l in range(len(dens)):
                        # prob = probMSD(MSDs[n], ks[n], (l + 1) * delta_R)
                        prob = prob_MSD_LUT[n, l]
                        #print("prob_MSD_LUT[n, l]", prob)
                        #print("n", n)
                        #print("l",l)
                        #print("m",m)
                        denominator += prob * dens[l] / sumpm
                        #print("denominator", denominator)
                        if np.isnan(denominator) :
                            return
                    sum_over_n += numerator / denominator
                dens[m] *= 1.0 / nb_track * sum_over_n


            newChiSquared = getChiSquared(dens)
            changeChiSquared = abs(newChiSquared - lastChiSquared) / lastChiSquared
            if np.isnan(changeChiSquared) :
                return
            lastChiSquared = newChiSquared


        # normalize density
        dens /= np.sum(dens)
        hist_x = np.zeros(len(dens))
        for i in range(len(dens)):
            hist_x[i] = bin_size_nm * (i + 1)   # To Diameter in [nm]
        #print("Xisquared",changeChiSquared)

        #print(hist_x,dens)
        plt.bar(hist_x,dens, width=5, edgecolor='black')
        plt.show()
        return hist_x, dens



    def get_stats(self):
        # Calculate some statistics
        mean_nspots, rcov, r_msd, r_gauss = [], [], [], []

        for track in self.tracks:
            if not track.is_filtered:
                mean_nspots.append(track.nSpots)
                rcov.append(track.r_cov)
                r_msd.append(track.r_msd)
                r_gauss.append(track.r_gauss)

        self.mean_nspots = np.mean(mean_nspots)
        self.mean_rcov = np.mean(rcov)
        self.mean_r_msd = np.mean(r_msd)
        self.mean_r_gauss = np.mean(r_gauss)
        self.std_rcov = np.std(rcov)
        self.std_r_msd = np.std(r_msd)
        self.std_r_gauss = np.std(r_gauss)
        self.median_rcov = np.median(rcov)
        self.median_r_msd = np.median(r_msd)
        self.median_r_gauss = np.median(r_gauss)

    def measure_concentration(self):
        """
        Adapter les idées de Marius à ce sujet.
        """
        pass

    def load_VideoData(self, params):
        """
        :param params:
        :return:
        """
        filmpath = params["filepath_video"]
        self.current_task = "Loading video in RAM"

        with tifffile.TiffFile(filmpath) as tif:
            #TODO gerer les erreurs d'ouverture du fichier
            # Extract frames from pages and convert to NumPy array
            self.video_array = np.stack([page.asarray() for page in tif.pages], axis=-1)


        # FIXME we should keep these information somewhere. Right ?
        height, width, frames = self.video_array.shape[:3]

        # Extract color channels
        self.video_array_red = self.video_array[:, :, 0, :]
        self.video_array_green = self.video_array[:, :, 1, :]
        self.video_array_blue = self.video_array[:, :, 2, :]    #FIXME not tested

        # Update the frames structure where spot are organized by time
        # All we need to do is to update their raw image data
        for video_frame_num in range(frames):
            self.frames[video_frame_num].img = [self.video_array_red, self.video_array_green, self.video_array_blue]
            


        self.current_task = "Analyze momomere"
        self.Analyze_color()
        self.get_ratio_color()

        self.current_task = "Idle"

        #if not film.isOpened():
            #TODO Log
        #    print("Error opening video.")
        #    return


    def calculate_draw_tracks(self,track_number):
        """
        TODO
        :param track_number:
        :return:
        """
        if track_number != " ":
            track_number = int(float(track_number))
            if 0 < track_number < np.size(self.tracks,0) :
                track = self.tracks[track_number]
                track_x = track.x
                track_y = track.y

        else :
            #print("track doesnt exist")
            track_x = []
            track_y = []
        return track_x,track_y


    def calculate_distance(self, frame_number, threshold):
        """
        :param frame_number:
        :param threshold:
        :return:
        """
        position = []
        for track in self.tracks:
            index = np.where(track.t == frame_number)[0]
            if index.size > 0:
                x = track.x[index[0]]
                y = track.y[index[0]]
                position.append((x, y))
        for l in range(len(position)) :
            for j in range(len(position)) :
                if l != j :
                    distance = np.sqrt((position[j][0]-position[l][0])**2 + (position[j][1]-position[l][1])**2)
                    #print(distance)
                    if distance < threshold :
                        self.overlap += 1
                        self.overlap_time.append(frame_number)


    def calculate_box_position(self, frame_number):
        """
        TODO ?
        :param frame_number:
        :return:
        """
        position = []
        tracker_color = []
        if self.tracks != None :
            for track in self.tracks :
                index = np.where(track.t  == frame_number)[0]
                if index.size > 0 :
                    #if track.x[index] != None:
                    x = track.x[index[0]]
                    #if track.y[index] != None:
                    y = track.y[index[0]]
                    if self.video_array_red[int(y),int(x),frame_number] > self.video_array_green[int(y),int(x),frame_number] :
                        tracker_red = 1
                    else :
                        tracker_red = 0

                    tracker_color.append(tracker_red)
                    position.append((x, y))
            #print(position)


        return position,tracker_color

    def Analyze_color(self):#, params):
        """
        :return:
        """

        self.box_radius = 4 #params["Analyze_particle_box_size_in_pixel"]    # ? FIXME Hardcoded
        self.red = 0
        self.green = 0

        for track in self.tracks:
            # For each spot in one track, we calculate the value of the red and green canal around the particle in a size self.
            track.spot_colors = []
            track.spot_main_color = []
            Value_red = 0
            Value_green = 0
            Value_blue = 0
            for m in range(np.size(track.x)):
                #FIXME pourquoi -1 sur le compteur temps ?
                Value_red += np.sum(self.video_array_red[int(track.y[m]) - self.box_radius:int(track.y[m]) + self.box_radius, int(track.x[m]) - self.box_radius:int(track.x[m]) + self.box_radius, int(track.t[m]) - 1])
                Value_green += np.sum(self.video_array_green[int(track.y[m]) - self.box_radius:int(track.y[m]) + self.box_radius, int(track.x[m]) - self.box_radius:int(track.x[m]) + self.box_radius, int(track.t[m]) - 1])
                Value_blue += np.sum(self.video_array_blue[int(track.y[m]) - self.box_radius:int(track.y[m]) + self.box_radius,int(track.x[m]) - self.box_radius:int(track.x[m]) + self.box_radius, int(track.t[m]) - 1])


            color = [1*Value_red, 1*Value_green, Value_blue]
            track.color_tracker = color
            #track.spot_colors.append(color)

            #track.spot_main_color.append(np.argmax(color))   # 0 for red, 1 for green, 2 for blue

            # Attribute color to the track -> Among all spots, what color was the most present.
            def most_frequent_value(lst):
                values, counts = np.unique(lst, return_counts=True)
                most_frequent_index = np.argmax(counts)
                return values[most_frequent_index]
            #track.color = most_frequent_value(track.spot_main_color)
            track.color = color.index(max(color))

    def get_ratio_color(self):
        self.r, self.g, self.b = 0,0,0
        self.green_radius,self.red_radius,self.blue_radius =[],[],[]
        for track in self.tracks:
            if not track.is_filtered:
                if track.color == 0:
                    self.r += 1

                elif track.color == 1:
                    self.g += 1

                elif track.color == 2:
                    self.b += 1

        total = self.r + self.g + self.b
        self.ratio_red = self.r*100 / total
        self.ratio_green = self.g*100 / total
        self.ratio_blue = self.b*100 / total



    def exportData(self, filename):
        data = (self.Moyenner)
        np.savetxt(filename, data,newline='\n')


    def get_all_delta_in_chronological_order(self):
        pass


    def generate_brownian_track(self, params_dict):

        self.space_units_nm = params_dict["space_units_nm"]
        self.T = params_dict["T"]
        self.eta = params_dict["eta"]
        self.frame_rate = 1 / params_dict["FrameRate"]

        particle_1_mean_diam_nm = params_dict["d1_nm"]
        particle_1_diam_sigma_relative = params_dict["sigma1_percent"]
        nb_particle_1 = int(params_dict["nb_particle_1"])
        particle_2_mean_diam_nm = params_dict["d2_nm"]
        particle_2_diam_sigma_relative = params_dict["sigma2_percent"]
        nb_particle_2 = int(params_dict["nb_particle_2"])

        nb_spot_per_track = int(params_dict["nb_spot_per_track"])


        #FIXME PARAMS ?
        dim_box_X_micron = 3000
        dim_box_Y_micron = 3000
        dim_box_Z_micron = 50

        #depth_of_focus_micron = 100
        drift_X_microns_per_frame = 0
        drift_Y_microns_per_frame = 0

        #FIXME explain
        #loc_values = np.random.choice([particle_1_mean_diam_nm, particle_2_mean_diam_nm], size=nb_particle,p=[ratio_monomere/100,(100-ratio_monomere)/100])

        kb = 1.380649E-23

        r1 = particle_1_mean_diam_nm/2
        sigma_1 = r1 * particle_1_diam_sigma_relative/100
        particle_radiuss_1 = np.random.normal(loc=r1, scale=sigma_1, size=nb_particle_1)

        r2 = particle_2_mean_diam_nm / 2
        sigma_2 = r1 * particle_2_diam_sigma_relative/100
        particle_radiuss_2 = np.random.normal(loc=r2, scale=sigma_2, size=nb_particle_2)

        particle_radiuss = np.concatenate((particle_radiuss_1, particle_radiuss_2))
        nb_particle = nb_particle_1 + nb_particle_2

        #mean_diff_coeff = kb * T / (6*np.pi*eta*(particle_1_mean_diam_nm/2*1E-9))
        diff_coeffs = kb * self.T / (6 * np.pi * self.eta * (particle_radiuss*1E-9))

        # brownian_length = np.sqrt(2*mean_diff_coeff*delta_t_ms*1E-3)
        # mean_nb_spot_pert_track  = 0
        # mean_dwell_time_in_focus_s = (depth_of_focus_micron*1E-9)**2/(2*mean_diff_coeff)

        npParticleType = np.dtype([('x', np.float64), ('y', np.float64), ('z', np.float64), ('Dtx', np.float64), ('Dty', np.float64),
             ('Dtz', np.float64)])
        particles = np.zeros(nb_particle, dtype=npParticleType)

        # Same diffusion coefficient for all direction : the particle are isotropic
        particles[:]['Dtx'] = particles[:]['Dtz'] = particles[:]['Dty'] = diff_coeffs

        # Initial position -> offset to all position
        r = np.random.rand(nb_particle, 3)
        particles[:]['x'] = r[:, 0] * dim_box_X_micron
        particles[:]['y'] = r[:, 1] * dim_box_Y_micron
        particles[:]['z'] = r[:, 2] * 2*dim_box_Z_micron - dim_box_Z_micron # centered on z = 0 i.e. the focal plane

        #Brownian motion
        # Draw random samples from a normal (Gaussian) distribution.
        dr = np.random.normal(loc=0, scale=1.0, size=(nb_spot_per_track, nb_particle, 3))

        # Constant drift
        dr[:, :, 0] += drift_X_microns_per_frame
        dr[:, :, 1] += drift_Y_microns_per_frame

        # Construct the brownian trajectory by adding all the displacement
        dr = np.cumsum(dr, axis=0, out=dr)

        # TODO do not create a new array at each iteration
        mvt_evolution = np.zeros((nb_spot_per_track, nb_particle), dtype=npParticleType)

        # offsetting at t=0 by the initial position
        mvt_evolution[:] = particles

        # Scaling the displacement with the diffusion coefficient -> in µm
        mvt_evolution[:]['x'] += dr[:, :, 0] * np.sqrt(2 * particles[:]['Dtx'] * self.frame_rate) * 1E6
        mvt_evolution[:]['y'] += dr[:, :, 1] * np.sqrt(2 * particles[:]['Dty'] * self.frame_rate) * 1E6
        mvt_evolution[:]['z'] += 0 #dr[:, :, 2] * np.sqrt(2 * particles[:]['Dtz'] * self.timeUnits) * 1E6

        # Create track from brownian trajectory
        self.tracks = []


        xs = mvt_evolution[:]['x']
        ys = mvt_evolution[:]['y']
        zs = mvt_evolution[:]['z']
        #dof = depth_of_focus_micron
        pos_micrometer_to_pixel = 1 / (self.space_units_nm / 1000)

        for i in range(nb_particle):
            track = Track()
            track.f = list(range(1, nb_spot_per_track + 1))
            track.x = xs[:, i] * pos_micrometer_to_pixel
            track.y = ys[:, i] * pos_micrometer_to_pixel
            track.z = zs[:, i] * pos_micrometer_to_pixel
            track.quality = 1
            track.nSpots = track.x.size
            self.tracks.append(track)

            """
            # pour gerer un eventuel defocus
            def consecutive(data, stepsize=1):
                # return np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)
                idx = np.r_[0, np.where(np.diff(data) != stepsize)[0] + 1, len(data)]
                return [data[i:j] for i, j in zip(idx, idx[1:])]            
            z = zs[:, i]
            #idx_in_focus = np.where(np.logical_and(z < dof, z > -dof))
            if len(idx_in_focus[0]) > 1:  # Test if the list is not empty
                tracks_idx = consecutive(idx_in_focus)
                # FIXME there is a problem with the "consecutive" function that return empty array
                
                for j in range(len(tracks_idx)):
                    if len(tracks_idx[j]) > 0:
                        # Create new track
                        track = Track()
                        track.x = xs[tracks_idx[j], i][0] * pos_micro_to_pixel
                        track.y = ys[tracks_idx[j], i][0] * pos_micro_to_pixel
                        track.z = zs[tracks_idx[j], i][0]
                        track.t = tracks_idx[j][0]
                        track.quality = 1
                        track.nSpots = track.x.size
                        self.tracks.append(track)
            """

        self.nTracks = len(self.tracks)

        params = {
            "T": self.T,
            "eta": self.eta,
            "timeUnits": self.frame_rate,
            "space_units_nm": self.space_units_nm,
            "R": self.R,
            "algo_drift_compensation": self.algo_drift,
            "min_nb_spot_for_gaussian_fit": self.min_nb_spot_for_gaussian_fit,
            "min_nb_spot_for_covariance": self.min_nb_spot_for_covariance
        }

        #FIXME : multiprocessing
        for track in self.tracks:
            track.calculate_D_from_gauss(params)
            track.calculate_D_from_MSD(params)
            track.calculate_D_from_covariance(params)
            track.Tensor_brownian(params)

        self.get_stats()


    def filter_tracks(self, low1, high1, type1, not1, bool_op, low2, high2, type2, not2):
        #FIXME inclusive frontiere ?
        def is_to_be_filtered(track, type, low, high):
            filter_OK = False

            val = None
            if type == "nSpots":
                val = track.nSpots
            elif type == "r_gauss":
                val = track.r_gauss*1E9
            elif type == "r_msd":
                val = track.r_msd*1E9
            elif type == "r_cov":
                val = track.r_cov*1E9
            elif type == "red_mean":
                val = track.red_mean
            elif type == "green_mean":
                val = track.red_mean
            elif type == "asym" :
                val = float(track.asym)


            if val is None:
                return False

            if low is None:
                if high is None:
                    return False
                elif val > high:
                    filter_OK = True
            else:
                if high is None:
                    if val < low:
                        filter_OK = True
                else:
                    if val < low or val > high:
                        filter_OK = True

            return filter_OK


        for track in self.tracks:
            filter1_OK = is_to_be_filtered(track, type1, low1, high1)
            filter2_OK = is_to_be_filtered(track, type2, low2, high2)

            if not1 and type1 != "None":
                filter1_OK = not filter1_OK
            if not2 and type2 != "None":
                filter2_OK = not filter2_OK

            is_track_filtered = False
            if bool_op == "and":
                is_track_filtered = filter1_OK and filter2_OK
            elif bool_op == "or":
                is_track_filtered = filter1_OK or filter2_OK
            elif bool_op == "xor":
                is_track_filtered = (filter1_OK and not(filter2_OK)) or (not(filter1_OK) and filter2_OK)

            if is_track_filtered:
                track.is_filtered = True
            else:
                track.is_filtered = False

    def get_correlation_graph(self, data_1_type, data_2_type):
        x_axis = []
        y_axis = []
        track_ID = []
        color_list = []
        def get_data_from_type(track, type):
            if type == "nSpots":
                return track.nSpots
            elif type == "r_gauss":
                return track.r_gauss
            elif type == "r_msd":
                return track.r_msd
            elif type == "r_cov":
                return track.r_cov
            elif type == "red":
                return track.color_tracker[0]*100/(track.color_tracker[0]+track.color_tracker[1]+track.color_tracker[2])
            elif type == "green":
                return (track.color_tracker[1]+track.color_tracker[2])*100/(track.color_tracker[0]+track.color_tracker[1]+track.color_tracker[2])
            elif type == "blue":
                return track.color_tracker[2]*100/(track.color_tracker[0]+track.color_tracker[1]+track.color_tracker[2])
            elif type == "asym" :
                return float(track.asym)

        if data_1_type == data_2_type:
            data_histo = []
            for track in self.tracks:
                continue
                data_histo.append(get_data_from_type(track, data_1_type))


            # We have to create an histogramm
            hist, bin_edges = np.histogram(data_histo)
            return hist, bin_edges

        i=-1
        for track in self.tracks:
            if track.is_filtered:
                i= i+1
                continue
            i += 1
            x_axis.append(get_data_from_type(track, data_1_type))
            if track.color == 0:
                color = "Red"
            elif track.color == 1:
                color = "Green"
            elif track.color == 2:
                color = "Blue"
            else :
                color = "None"
            row = [
                i,
                int(track.f[0]),
                track.nSpots,
                np.round(track.r_cov*1e9,2),
                np.round(track.error_r_cov*1e9,2),
                color,
                float(track.asym),
                track.x,
                track.y]
            track_ID.append(row)
            y_axis.append(get_data_from_type(track, data_2_type))
            color_list.append(color)

        return x_axis, y_axis,track_ID,color_list

    def clear_filter(self):
        for track in self.tracks:
            track.is_filtered = False
        pass
    def filtered_click(self,track_info):

        track = self.tracks[track_info]
        track.is_filtered = True




    def get_full_MSD(self):
        msd = []
        for track in self.tracks:
            if track.is_filtered is not True:
                msd.append(track.get_full_MSD([track.x, track.y]))
        self.full_msd = np.mean(msd, axis=0)
        self.full_msd_x = np.arange(0, len(self.full_msd))*self.frame_rate
        return self.full_msd_x, self.full_msd

    def save_data_gauss(self,filePath):
        filename = filePath.name
        combined_array = np.column_stack((self.x_full_gauss, self.Gauss_full_track))
        with open(filename, 'w') as file:
            # Write header
            file.write("Radius (Nanometers)\tValue of the normalized Gaussian\n")

            # Write data
            for row in combined_array:
                file.write(f"{row[0]}\t{row[1]}\n")

    def save_data_hist(self,filePath):
        filename = filePath.name
        nb_bins = int(np.max(self.Moyenner)/5)
        Histoy, Histox = np.histogram(self.Moyenner, bins=nb_bins)
        #print(Histoy)
        #print(Histox)
        Histoy = np.append(Histoy, 0)
        combined_array = np.column_stack((Histox, Histoy))
        with open(filename, 'w') as file:
            # Write header
            file.write("Radius (Nanometers)\tOccurence\n")
            # Write data
            for row in combined_array:
                file.write(f"{row[0]}\t{row[1]}\n")

    def save_red_data_hist(self,filePath):
        filename = filePath.name
        nb_bins = int(np.max(self.Moyenner)/5)
        filtered_data = np.array(self.Moyenner)[self.number_tracks_red]
        nb_bins = int(np.max(filtered_data) / 5)
        Histoy, Histox = np.histogram(filtered_data, bins=nb_bins)
        #print(Histoy)
        #print(Histox)
        Histoy = np.append(Histoy, 0)
        combined_array = np.column_stack((Histox, Histoy))
        with open(filename, 'w') as file:
            # Write header
            file.write("Radius (Nanometers)\tOccurence\n")
            # Write data
            for row in combined_array:
                file.write(f"{row[0]}\t{row[1]}\n")

    def save_green_data_hist(self,filePath):
        filename = filePath.name
        nb_bins = int(np.max(self.Moyenner)/5)
        filtered_data = np.array(self.Moyenner)[self.number_tracks_green]
        nb_bins = int(np.max(filtered_data) / 5)
        Histoy, Histox = np.histogram(filtered_data, bins=nb_bins)
        #print(Histoy)
        #print(Histox)
        Histoy = np.append(Histoy, 0)
        combined_array = np.column_stack((Histox, Histoy))
        with open(filename, 'w') as file:
            # Write header
            file.write("Radius (Nanometers)\tOccurence\n")
            # Write data
            for row in combined_array:
                file.write(f"{row[0]}\t{row[1]}\n")

    def save_state(self, filename):
        """
        Save the state of the object to a file using shelve.
        """
        print(filename)
        with shelve.open(filename, 'n') as db:
            for attr, value in self.__dict__.items():
                db[attr] = value

    def load_state(self, filename):
        """
        Load the state of the object from a file using shelve.
        """
        with shelve.open(filename, 'o') as db:
            for attr, value in db.items():
                setattr(obj, attr, value)

    def get_data_from_num_track(self, num):
        return self.tracks[num]

    def get_x_data(self, num):
        return self.tracks[num].x

    def get_y_data(self, num):
        return self.tracks[num].y




if __name__ == "__main__":
    core = AnalyzeTrackCore()
    print(core.space_units_nm)
    core.generate_brownian_track(params_dict=None)
    pass


