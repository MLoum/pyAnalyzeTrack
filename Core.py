import matplotlib.pyplot as plt
import numpy as np
import xml.etree.cElementTree as et
from lmfit import Model
from scipy.fftpack import dst

#import pandas as pd
import os

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Value
import ctypes
import tifffile

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
        self.Dx_cov, self.Dy_cov = None, None
        self.var_Dx_cov, self.var_Dy_cov = None, None
        self.result_gauss_x, self.result_gauss_y = None, None

        self.red_mean, self.green_mean = 0, 0

        self.r_gauss, self.r_cov_2 = 0, 0
        self.error_r_gauss, self.error_r_cov_2, self.error_r_msd = 0, 0, 0

        self.is_filtered = False

    def calculate_displacement(self, pos, algo="None", step=1):
        """
        Return the list of all displacement Δx, defined as Δx = x[i+1] - x[i], while taking into account an eventual drift (cf algo)

        :param algo: Algorithm for drift compensation.
        "itself" remove the mean of the data from all displacements
        :return:
        """
        if algo == "None":
            return np.diff(pos,step)
        elif algo == "itself":
            return np.diff(pos,step) - np.mean(pos)
        elif algo == "neighbors":
            # TODO
            return np.diff(pos), np.diff(pos)

    def calculate_D_from_gauss(self, params):
        """
        TODO la variance et l'erreur statistique sur la valeur de D
        Var (Δx) = <Δx²> - (<Δx>)²
        <Δx²> est la MSD est on peut en déduire le coefficient de diffusion via  <Δx²> = 2 D Δt
        (<Δx>)² est le carré de la moyenne, on peut en déduire une estimation, c'est le centre de la gaussienne.
        Var (Δx) provient de l'ajustement de l'histogramme par une gaussienne.
        :return:
        """
        kb_ = 1.38E-23
        algo_drift_compensation = params["algo_drift_compensation"]
        min_nb_spot_for_gaussian_fit = params["min_nb_spot_for_gaussian_fit"]


        def gaussian_delta_fit(pos, algo_drift_compensation="None"):
            """
            :param pos: Tableau contenant les positions au cours du temps.
            :param algo_drift_compensation:
            :return:
            """

            def gaussian(x, amp, cen, wid):
                """
                Une gaussienne (non normalisée) est définie par :
                exp(-1/2 (x-µ)²/σ²

                où µ est la moyenne
                et σ est l'écart type (standart deviation) avec donc σ² <=> Variance.
                ici wid est égale à 2 σ²
                :param x:
                :param amp:
                :param cen: µ
                :param wid: 2 σ²
                :return:
                """
                return amp * np.exp(-(x - cen) ** 2 / wid)

            diff = self.calculate_displacement(pos, algo_drift_compensation)

            # https://indico.cern.ch/event/428334/contributions/1047042/attachments/919069/1299634/ChoosingTheRightBinWidth.pdf

            hist, boundaries = np.histogram(diff, bins="sturges")
            boundaries = boundaries[0:-1]
            gmodel = Model(gaussian, nan_policy='raise')
            max_, min_ = np.max(hist), np.min(hist)
            max_bound, min_bound = np.max(boundaries), np.min(boundaries)
            params = gmodel.make_params(cen=np.mean(diff), amp=max_, wid=np.std(diff))
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

        def D_from_fit(center, width, params):
            T = params["T"]
            eta = params["eta"]
            timeUnits = params["timeUnits"]
            space_units_nm = params["space_units_nm"]

            mean_squared_SI = (center * space_units_nm * 1E-9) ** 2
            variance_SI = (width * space_units_nm * space_units_nm * 1E-18) / 2
            #FIXME Cela me semble faux de ne pas prendre en compte la moyenne. On l'a peut-etre compenser, mais alors on doit avoir presque zero
            msd = variance_SI  # Or: msd = variance_SI + mean_squared_SI, depending on your needs
            D_gauss = msd / (2 * timeUnits)
            r_gauss = kb_ * T / (6 * np.pi * eta * D_gauss)

            return D_gauss, r_gauss

        if self.nSpots < min_nb_spot_for_gaussian_fit:
            self.Dx_gauss = -100
            self.Dy_gauss = -100
            self.r_gauss = -1E6
            return


        diff, hist, boundaries, result, center, width = gaussian_delta_fit(self.x, algo_drift_compensation)
        # Test if the fit converged by looking for instance at the first returned value
        if diff is not None:
            self.Dx_gauss, rx = D_from_fit(center.value, width.value, params)
        else:
            # A big negative number
            self.Dx_gauss = -100
            rx = -1E6

        diff, hist, boundaries, result, center, width = gaussian_delta_fit(self.y, algo_drift_compensation)
        # Test if the fit converged by looking for instance at the first returned value
        if diff is not None:
            self.Dy_gauss, ry = D_from_fit(center.value, width.value, params)
        else:
            # A big negative number
            self.Dy_gauss = -100
            ry = -1E6

        self.r_gauss = (rx + ry)/2


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
                                           |- Due to the error of localisation σ and motion blur (R : motion blur coefficien)

        < Δx_n . Δx_(n+1)> = - (σ² − 2.D.R.Δt)  NB : Non zero covariance. Should be 0 for a memoryless perfect Brownian motion.
        We can inverse these equations to find :
                <(Δx_n)²> - 2 . σ²
        D = ---------------------------
                2 . (1 - 2.R) . Δt

        σ² = R <(Δx_n)²> + (2.R - 1) . < Δx_n . Δx_(n+1)>

        with the variance on D equals to :

                       2 ∈² + 4 ∈ + 6     4 (1 + ∈²)
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
            drift = np.nanmean(diff)
            MSD = np.mean((diff - drift) ** 2)

            # covariance = np.mean((diff[:-1]) * (diff[1:]))
            covariance = np.mean((diff[:-1] - drift) * (diff[1:] - drift))

            sigma_square = R * MSD + (2 * R - 1) * covariance
            D_cov = (MSD - 2 * sigma_square) / ((2 - 4 * R) * timeUnits)
            if D_cov == 0:
                return -1, -1, -1
            r_cov_2 = (kb_ * T) / (6 * np.pi * eta * D_cov)

            epsilon = sigma_square / (D_cov * timeUnits) - 2 * R
            N = self.nSpots
            var_D_cov = D_cov**2 * ((6 + 4 * epsilon + 2 * epsilon ** 2) / N + (4 * (1 + epsilon ** 2)) / (N ** 2))

            return D_cov, r_cov_2, var_D_cov

        if self.nSpots < min_nb_spot_for_covariance:
            self.Dx_cov, self.Dy_cov, self.r_cov_2 = -1, -1, -1
            return

        #TODO error on R
        self.Dx_cov, rx_cov, self.Dx_cov = caculate_cov_1D(self.x, params)
        self.Dy_cov, ry_cov, self.Dy_cov = caculate_cov_1D(self.y, params)
        self.r_cov_2 = (rx_cov + ry_cov)/2

    def extract_colored_ROI(self, video_frames, color):

        # NB : Vectorized
        # Keep only the frame corresponding to the track
        if video_frames is None:
            # No movie associated with the track. We keep the defulat values
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

"""
En dehors des class pour pouvoir le sérialiser lors des processus multicore
"""
def init_track(data, params):
    """
    Using ProcessPoolExecutor has the disadvantage that each child process runs in its own memory space,
    separate from the parent process. This means that any changes made to an object in the child process
    will not be reflected in the parent process.
    Solution:
    Return the modified object:  return the modified object from the function you're mapping,
    and then use these returned objects to update the list in the parent process.

    :param data:
    :return:
    """
    track = Track()
    #TODO the value of the columns are "magic numbers", we should identify them in the header instead of relying on a given order
    track.quality = data[:, 2]
    track.x = data[:, 3]
    track.y = data[:, 4]
    track.z = data[:, 5]
    track.t = data[:, 6]
    track.r_trackmate = data[:, 8] # This is the "RADIUS" column, the value is fixed and correspond to the one given by the user during the spot detection.
    # track.r_trackmate = data[:, 17] # This is ESTIMATED_DIAMETER, should be better.
    #TODO Add contrast and/or SNR.

    track.nSpots = track.x.size

    track.calculate_D_from_gauss(params)
    track.calculate_D_from_MSD(params)
    track.calculate_D_from_covariance(params)

    #global task_counter
    #task_counter += 1
    #print("task_counter = ", task_counter)

    return track

class AnalyzeTrackCore():
    """

    """
    def __init__(self):
        self.Filter = 10
        self.overlap = 0
        self.overlap_time = []
        self.min_nb_spot_for_gaussian_fit = 10  #FIXME Hardcoded
        self.min_nb_spot_for_covariance = 10  # FIXME Hardcoded
        self.tracks = None
        self.nTracks = None
        self.frameInterval = None
        self.space_units_nm = 172.5 # in nm
        self.timeUnits = 1/120 # in s
        self.T = 293    #in K
        self.eta = 0.001    # in Pa.s
        self.sigma_already_known = False
        self.R = 1/6
        self.division = 10
        self.tracker = 0

        self.is_computing = False

        self.executor = None

        self.algo_drift = "itself"

        # Tracking of the calculations progresses
        self.tasks_completed = Value(ctypes.c_int, 0)
        self.current_task = "Idle"
        self.task_counter = 0


    def calculate_displacement(self, pos, algo="None", step=1):
        """
        Return the list of all displacement Δx, defined as Δx = x[i+1] - x[i], while taking into account an eventual drift (cf algo)

        :param algo: Algorithm for drift compensation.
        "itself" remove the mean of the data from all displacements
        :return:
        """
        if algo == "None":
            return np.diff(pos,step)
        elif algo == "itself":
            return np.diff(pos,step) - np.mean(pos)
        elif algo == "neighbors":
            # TODO
            return np.diff(pos), np.diff(pos)



    def viscosity_from_temperature(self, solvent="water", T=293):
        # TODO tabuler la viscosité de l'eau.
        if solvent == "water":
            return 1E-3

        return None

    def change_filter(self, Valeur):
        self.Filter = int(Valeur.get())

    def get_sublists(self,original_list, sublists_number):
        list_size = len(original_list)
        sublist_size_except_last = list_size // sublists_number
        last_sublist_size = list_size % sublists_number
        sublists = list()
        l_index = 0
        r_index = list_size - 1

        for i in range(sublists_number):
            l_index = (i * sublist_size_except_last)
            r_index = ((i + 1) * sublist_size_except_last) - 1
            if i != sublists_number - 1:
                sublists.append(original_list[l_index:r_index + 1])
            else:
                r_index = r_index + last_sublist_size
                sublists.append(original_list[l_index:r_index + 1])

        return sublists
    def initialize_variables(self):
        """ We initialize all the variables needed to calculate all the parameter with the covariance 
        method. This will be useful to reset the software without having to close it.
        """
        self.taille = []
        self.Spectre = []
        self.Spectre_mean = []
        self.Spectre_moyenne = []
        self.Moyennex = 0
        self.Moyenney = 0
        self.i = 0
        self.alpha = 0
        self.beta = 0
        self.Moyenner = []
        self.Controle = []
        self.Controle_variance = []
        self.Controle_track = np.array([])
        self.lim_min = 0 * 10 ** -9
        self.lim_max = 200 * 10 ** -9
        self.nombre = int((self.lim_max * 10 ** 9 - self.lim_min * 10 ** 9) + 1)
        self.Moyenner = []
        self.Gauss1 = []
        self.Gauss = []
        self.MSDx_lag = []
        self.covariancex_lag = []
        self.trivariancex_lag = []
        self.quadravariancex_lag = []
        self.pentavariancex_lag = []
        self.sixtvariancex_lag = []
        self.septavariancex_lag = []
        self.octavariancex_lag = []
        self.MSDx_lag_variance = []
        self.covariancex_lag_variance = []
        self.trivariancex_lag_variance = []
        self.quadravariancex_lag_variance = []
        self.pentavariancex_lag_variance = []
        self.sixtvariancex_lag_variance = []
        self.septavariancex_lag_variance = []
        self.octavariancex_lag_variance = []
        self.MSDy_lag = []
        self.covariancey_lag = []
        self.trivariancey_lag = []
        self.quadravariancey_lag = []
        self.pentavariancey_lag = []
        self.sixtvariancey_lag = []
        self.septavariancey_lag = []
        self.octavariancey_lag = []
        self.x_full_gauss = [i for i in np.linspace(self.lim_min*10**9, self.lim_max*10**9, self.nombre)]


    def calculate_lag(self, diff_x_1,diff_y_1,track):

        """ For each track we can get the lags which is the mean displacement between 2 or more frame. If the movement
        is perfectly brownien the lags should be 0 for all lags superior to 2 ( we have to take account of the experimental error obviously a variance
         is then added to each result)"""

        self.MSDx_lag.append(np.mean((diff_x_1) * (diff_x_1)))
        self.covariancex_lag.append(np.mean((diff_x_1[:-1]) * (diff_x_1[1:])))
        self.trivariancex_lag.append(np.mean((diff_x_1[:-2]) * (diff_x_1[2:])))
        self.quadravariancex_lag.append(np.mean((diff_x_1[:-3]) * (diff_x_1[3:])))
        self.pentavariancex_lag.append(np.mean((diff_x_1[:-4]) * (diff_x_1[4:])))
        self.sixtvariancex_lag.append(np.mean((diff_x_1[:-5]) * (diff_x_1[5:])))
        self.septavariancex_lag.append(np.mean((diff_x_1[:-6]) * (diff_x_1[6:])))
        self.octavariancex_lag.append(np.mean((diff_x_1[:-7]) * (diff_x_1[7:])))
    
        self.MSDx_lag_variance.append(0)
        self.covariancex_lag_variance.append(0)
        self.trivariancex_lag_variance.append(((self.alpha + 4 * self.alpha * self.beta + 6 * self.beta ** 2) / (track.nSpots - 2)) - ((2 * self.beta ** 2) / ((track.nSpots - 2) ** 2)))
        self.quadravariancex_lag_variance.append(((self.alpha + 4 * self.alpha * self.beta + 6 * self.beta ** 2) / (track.nSpots - 3)) - ((2 * self.beta ** 2) / ((track.nSpots - 3) ** 2)))
        self.pentavariancex_lag_variance.append(((self.alpha + 4 * self.alpha * self.beta + 6 * self.beta ** 2) / (track.nSpots - 4)) - ((2 * self.beta ** 2) / ((track.nSpots - 4) ** 2)))
        self.sixtvariancex_lag_variance.append(((self.alpha + 4 * self.alpha * self.beta + 6 * self.beta ** 2) / (track.nSpots - 5)) - ((2 * self.beta ** 2) / ((track.nSpots - 5) ** 2)))
        self.septavariancex_lag_variance.append(((self.alpha + 4 * self.alpha * self.beta + 6 * self.beta ** 2) / (track.nSpots - 6)) - ((2 * self.beta ** 2) / ((track.nSpots - 6) ** 2)))
        self.octavariancex_lag_variance.append(((self.alpha + 4 * self.alpha * self.beta + 6 * self.beta ** 2) / (track.nSpots - 7)) - ((2 * self.beta ** 2) / ((track.nSpots - 7) ** 2)))
    
        self.MSDy_lag.append(np.mean((diff_y_1) * (diff_y_1)))
        self.covariancey_lag.append(np.mean((diff_y_1[:-1]) * (diff_y_1[1:])))
        self.trivariancey_lag.append(np.mean((diff_y_1[:-2]) * (diff_y_1[2:])))
        self.quadravariancey_lag.append(np.mean((diff_y_1[:-3]) * (diff_y_1[3:])))
        self.pentavariancey_lag.append(np.mean((diff_y_1[:-4]) * (diff_y_1[4:])))
        self.sixtvariancey_lag.append(np.mean((diff_y_1[:-5]) * (diff_y_1[5:])))
        self.septavariancey_lag.append(np.mean((diff_y_1[:-6]) * (diff_y_1[6:])))
        self.octavariancey_lag.append(np.mean((diff_y_1[:-7]) * (diff_y_1[7:])))


    def Lag_all_track(self):

        """ Once we get the staistical analysis for each track we can create an analysis of the whole film which get rid of all the errors
        that can happen for short tracks for example to give us a better insight of the whole experiment"""

        self.Controle_track = self.MSDx_lag
        self.Controle_track = np.vstack((self.Controle_track, self.covariancex_lag))
        self.Controle_track = np.vstack((self.Controle_track, self.trivariancex_lag))
        self.Controle_track = np.vstack((self.Controle_track, self.quadravariancex_lag))
        self.Controle_track = np.vstack((self.Controle_track, self.pentavariancex_lag))
        self.Controle_track = np.vstack((self.Controle_track, self.sixtvariancex_lag))
        self.Controle_track = np.vstack((self.Controle_track, self.septavariancex_lag))
        self.Controle_track = np.vstack((self.Controle_track, self.octavariancex_lag))

        self.Controle_track_variance = self.MSDx_lag_variance
        self.Controle_track_variance = np.vstack((self.Controle_track_variance, self.covariancex_lag_variance))
        self.Controle_track_variance = np.vstack((self.Controle_track_variance, self.trivariancex_lag_variance))
        self.Controle_track_variance = np.vstack((self.Controle_track_variance, self.quadravariancex_lag_variance))
        self.Controle_track_variance = np.vstack((self.Controle_track_variance, self.pentavariancex_lag_variance))
        self.Controle_track_variance = np.vstack((self.Controle_track_variance, self.sixtvariancex_lag_variance))
        self.Controle_track_variance = np.vstack((self.Controle_track_variance, self.septavariancex_lag_variance))
        self.Controle_track_variance = np.vstack((self.Controle_track_variance, self.octavariancex_lag_variance))

        self.Controle.append(np.mean(self.MSDx_lag))
        self.Controle.append(np.mean(self.covariancex_lag))
        self.Controle.append(np.mean(self.trivariancex_lag))
        self.Controle.append(np.mean(self.quadravariancex_lag))
        self.Controle.append(np.mean(self.pentavariancex_lag))
        self.Controle.append(np.mean(self.sixtvariancex_lag))
        self.Controle.append(np.mean(self.septavariancex_lag))
        self.Controle.append(np.mean(self.octavariancex_lag))

        self.Controle_variance.append(np.mean(self.MSDx_lag_variance))
        self.Controle_variance.append(np.mean(self.covariancex_lag_variance))
        self.Controle_variance.append(np.mean(self.trivariancex_lag_variance))
        self.Controle_variance.append(np.mean(self.quadravariancex_lag_variance))
        self.Controle_variance.append(np.mean(self.pentavariancex_lag_variance))
        self.Controle_variance.append(np.mean(self.sixtvariancex_lag_variance))
        self.Controle_variance.append(np.mean(self.septavariancex_lag_variance))
        self.Controle_variance.append(np.mean(self.octavariancex_lag_variance))

    def process_track(self,track):
        """
        For each track we compute the deplacement of each particule for each frame. Thanks to that we can deduct the
        diffusion coefficient of the brownian movement which give us the size of the particule. We account for the experimentals errors like
        a convection motion in the solution or the errors created by the acquisition of the frames.

        We separate the X axis and the Y axis for each particle which in theory should have no difference since the particles rotates,
        even if the particule isnt spherical the movement on each axes is blended together in the long term.

        We also use the displacement to compute the lags and the covariance of each track, this indicate if the track is close to what would be expected
        of a perfect brownian motion which is supposed to have no memory between each displacement. Of course for each analysis we compute the error.


        :param track:
        :return:
        """
        
        self.i=self.i+1

        diff_x_1 = self.calculate_displacement(track.x * self.space_units_nm * 10 ** -9, algo="None")
        diff_y_1 = self.calculate_displacement(track.y * self.space_units_nm * 10 ** -9, algo="None")
        driftx = np.nanmean(diff_x_1)

        MSDx = np.mean((diff_x_1 - driftx) ** 2)
        covariancex = np.mean((diff_x_1[:-1] - driftx) * (diff_x_1[1:] - driftx))

        self.calculate_lag(diff_x_1,diff_y_1,track)
        
        sigma_squarex = self.R * MSDx + (2 * self.R - 1) * covariancex
        track.D_x = (MSDx - 2 * sigma_squarex) / ((2 - 4 * self.R) * self.timeUnits)
        track.x_cov = (1.38 * 10 ** (-23) * self.T) / (6 * np.pi * self.eta * track.D_x)
        self.alpha = 2 * track.D_x * self.timeUnits
        self.beta = sigma_squarex - 2 * track.D_x * self.timeUnits * self.R
        self.Spectre.append((2 * (0.5 * self.timeUnits * dst(diff_x_1, type=1)) ** 2) / ((track.nSpots + 1) * self.timeUnits * track.D_x * (self.timeUnits) ** 2))

        x1 = self.get_sublists(self.Spectre[self.i - 1], self.division)
        self.taille.append(np.size(x1[0]))
        for j in range(self.division):
            self.Spectre_moyenne = np.append(self.Spectre_moyenne, np.mean(x1[j]))

        self.Spectre_mean.append(self.Spectre_moyenne)
        self.Spectre_moyenne = []

        if np.abs(track.x_cov) > 200 * 10 ** -7:
            track.x_cov = 0
        if track.x_cov == np.nan:
            self.Moyennex = self.Moyennex
        else:
            self.Moyennex = self.Moyennex + track.x_cov



        drifty = np.nanmean(diff_y_1)
        MSDy = np.mean((diff_y_1 - drifty) ** 2)
        covariancey = np.mean((diff_y_1[:-1] - drifty) * (diff_y_1[1:] - drifty))

        sigma_squarey = self.R * MSDy + (2 * self.R - 1) * covariancey
        track.D_y = (MSDy - 2 * sigma_squarey) / ((2 - 4 * self.R) * self.timeUnits)
        track.y_cov = (1.38 * 10 ** (-23) * self.T) / (6 * np.pi * self.eta * track.D_y)

        if np.abs(track.y_cov) > 200 * 10 ** -7:
            track.y_cov = 0
        if track.y_cov == np.nan:
            self.Moyenney = self.Moyenney
        else:
            self.Moyenney = self.Moyenney + track.y_cov

        # track.r_cov = np.sqrt(track.x_cov**2+track.y_cov**2)
        # track.r_cov = track.y_cov
        track.r_cov = track.x_cov

        self.Moyenner.append(track.x_cov * 10 ** 9)
        # self.Moyenner.append(track.y_cov*10**9)

        zetha = sigma_squarex / (track.D_x * self.timeUnits) - 2 * self.R

        Variancex_D = track.D_x ** 2 * ((6 + 4 * zetha + 2 * zetha ** 2) / (track.nSpots) + (4 * (1 + zetha) ** 2) / (track.nSpots ** 2))
        track.x_cov_1 = (1.38 * 10 ** (-23) * self.T) / (6 * np.pi * self.eta * (track.D_x + np.sqrt(Variancex_D)))
        track.x_cov_2 = (1.38 * 10 ** (-23) * self.T) / (6 * np.pi * self.eta * (track.D_x - np.sqrt(Variancex_D)))

        Variance = ((track.x_cov_2 - track.x_cov_1) / 2) ** 2
        # Variance = (track.x_cov / track.D_x ) * Variance_D
        # Variance = np.sqrt(np.sqrt(MSDx ** 2 + MSDy ** 2) / (track.nSpots - 1))
        if track.x_cov != 0:
            self.Gauss1.append((1 / (np.sqrt(Variance * 10 ** 18) * np.sqrt(2 * np.pi))) * np.exp(-(self.x_full_gauss - track.x_cov * 10 ** 9) ** 2 / (2 * Variance * 10 ** 18)))

        track.error_r_cov = np.absolute((track.x_cov_2 - track.x_cov_1) / 2)
        track.error_r_cov = "{:.3e}".format(track.error_r_cov)
        track.r_cov = "{:.3e}".format(track.r_cov)
        #print(track.r_cov)
        
        
    def covariance(self,sigma_square=None, algo_drift="self"):
        """
        Where the program compute the covariance method.
        AFter the calculations we locate the most probable size of a particle after adding
        all of the particle size calculated normalized with the error of each computation
        """
        #self.tracker = 0
        self.initialize_variables()

        for track in self.tracks:
            self.task_counter += 1

            if track.nSpots < self.Filter:
                track.is_filtered = True
            if track.is_filtered is True:
                track.r_cov = np.nan
            else:
                self.process_track(track)

        self.Lag_all_track()
        self.Moyenne = np.nanmean(self.Moyenner)
        self.Gauss_full_track = np.nansum(self.Gauss1, axis=0)
        self.Gauss_full_track /= np.max(self.Gauss_full_track)
        Max_value = np.max(self.Gauss_full_track)
        self.Max_index = np.where(self.Gauss_full_track == Max_value)
        self.Max_index = self.Max_index[0] * 10 ** -9 + self.lim_min

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
        self.current_task = "Track Data loading"

        filepath = params["filepath_track"]

        def find_track_id_index(filepath):
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

            # Find the index of the "TRACK_ID" column
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
        # We don't take in account the first column which contains text and that is redundant with ID.
        # Typical Header :
        # Label    ID       TRACK_ID QUALITY POSITION_X POSITION_Y POSITION_Z POSITION_T FRAME RADIUS VISIBILITY MANUAL_COLOR MEAN_INTENSITY MEDIAN_INTENSITY MIN_INTENSITY MAX_INTENSITY TOTAL_INTENSITY STANDARD_DEVIATION ESTIMATED_DIAMETER CONTRAST SNR
        # ID10975  10975	0	     2.451	 317.287	192.140	0  0	0	5	1	-10921639	27.660	26	8	58	2683	10.804	5.096	0.279	1.116
        raw = np.loadtxt(filepath, skiprows=1, usecols=np.arange(1, num_columns-1))

        # Since we don't take in account Label, we need to update the Track_ID position
        track_id_index -= 1
        position_T_index -= 1

        nb_lines = np.shape(raw)[0]
        # print(nb_lines)

        # Tracks are listed by ascending trackID and then chronologicaly (MOST of the time, at some point, the track where temporarlily mixed)..


        # We search for the positions in the raw numpy array of each unique track based on their Track_ID
        unique, unique_index = np.unique(raw[:, track_id_index], return_index=True)
        self.nTracks = np.size(unique) - 1
        self.tracks = [None] * self.nTracks  # Create an empty list to store Track objects

        # Split the raw data into a list of sub-arrays, one per unique track
        data_splits = [raw[unique_index[i]:unique_index[i + 1], :] for i in range(self.nTracks)]
        #self.tracks = data_splits



        global tasks_completed
        tasks_completed = 0


        params = {
            "T": self.T,
            "eta": self.eta,
            "timeUnits": self.timeUnits,
            "space_units_nm": self.space_units_nm,
            "R": self.R,
            "algo_drift_compensation": self.algo_drift,
            "min_nb_spot_for_gaussian_fit": self.min_nb_spot_for_gaussian_fit,
            "min_nb_spot_for_covariance": self.min_nb_spot_for_covariance
        }

        # Create a ProcessPoolExecutor and populate the Track objects in parallel
        num_cores = os.cpu_count()

        with ProcessPoolExecutor(max_workers=int(num_cores*0.75)) as executor:
            self.tracks = list(executor.map(init_track, data_splits, repeat(params)))

        self.covariance()
        #for data in data_splits:
        #    self.tracks.append(init_track(data, params))

        #self.covariance()






        # #TODO Test Chronological order.
        # # Chronological order :
        # # Serach for all lines that shares the same time t
        # test = raw[:, 6]
        # unique, unique_index, unique_inverse, unique_counts = np.unique(raw[:, 6], return_index=True, return_inverse= True, return_counts=True)
        # self.nb_track_per_frame = unique_counts
        # self.id_of_track_at_time_t = None

        self.current_task = "idle"

    def load_VideoData(self, params):
        filmpath = params["filepath_video"]
        self.current_task = "Loading video in RAM"

        with tifffile.TiffFile(filmpath) as tif:
            # Extract frames from pages and convert to NumPy array
            self.video_array = np.stack([page.asarray() for page in tif.pages], axis=-1)

        height, width, frames = self.video_array.shape[:3]

        # Extract color channels
        self.video_array_red = self.video_array[:, :, 0, :]
        self.video_array_green = self.video_array[:, :, 1, :]

        self.Analyze_monomere()

        self.current_task = "Idle"

        #if not film.isOpened():
            #TODO Log
        #    print("Error opening video.")
        #    return


        """
        width = int(film.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(film.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames = int(film.get(cv2.CAP_PROP_FRAME_COUNT))

            # Initialize a 4D NumPy array to store the frames
        self.video_array = np.zeros((height, width, 3, frames), dtype=np.uint8)

        for i in range(frames):
            #print(i)
            ret, frame = film.read()
            if not ret:
                break
            self.video_array[:, :, :, i] = frame

        # Initialize a 4D NumPy array to store the frames
        self.video_array_red = np.zeros((height, width, frames), dtype=np.uint8)

            # Read and store each frame in the array
        for i in range(frames):
            self.video_array_red[:, :, i] = self.video_array[:, :, 2, i]

            # Initialize a 3D NumPy array to store the frames
        self.video_array_green = np.zeros((height, width, frames), dtype=np.uint8)

            # Read and store each frame in the array
        for i in range(frames):
            self.video_array_green[:, :, i] = self.video_array[:, :, 1, i]
        film.release()
        """


    def calculate_draw_tracks(self,track_number):
        track_number = int(float(track_number))
        if 1 < track_number < np.size(self.tracks,0) :
            track = self.tracks[track_number - 1]
            track_x = track.x
            track_y = track.y

        else :
            print("track doesnt exist")
        return track_x,track_y


    def calculate_distance(self,frame_number,threshold):
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

    def Analyze_monomere(self):

        self.Rayon_boite = 4
        self.red = 0
        self.green = 0
        self.compteur_red = 0
        self.compteur_green = 0


        for track in self.tracks:
            for m in range(np.size(track.x)) :
                Value_red = np.sum(self.video_array_red[int(track.y[m]) - self.Rayon_boite:int(track.y[m]) + self.Rayon_boite,int(track.x[m]) - self.Rayon_boite:int(track.x[m]) + self.Rayon_boite,int(track.t[m]) - 1])
                Value_green = np.sum(self.video_array_green[int(track.y[m]) - self.Rayon_boite:int(track.y[m]) + self.Rayon_boite,int(track.x[m]) - self.Rayon_boite:int(track.x[m]) + self.Rayon_boite,int(track.t[m]) - 1])
                if Value_red > Value_green:
                    self.red += 1
                else:
                    if Value_red < Value_green:
                        self.green += 1

                #print("Value_red = ", Value_red)
                #print("Value_green", Value_green)

            #print("red = ", red)
            #print("green =", green)

            #print("size=",np.size(track.x))
            #print("stop")
            if self.red > self.green:
                self.compteur_red += 1
            else:
                if self.red < self.green:
                    self.compteur_green += 1
                if self.red == self.green == 0:
                    Filtre = 1

            self.red = 0
            self.green = 0
        self.ratio = 100* self.compteur_green / (self.compteur_red + self.compteur_green)
        #print("red =", self.compteur_red)
        #print("green =", self.compteur_green)
        #print("ratio de monomère =", np.round(self.ratio , 4), "%")
        #print("Filtre =", Filtre)


        # TODO faire du multiprocessing, mais le partage de la mémoire est compliqué en python
        # Si on ne fait pas attention on se retrouve à copier les films entier pour chaque workers.
        # Il faut convertir en " multiprocessing  Array"
        #self.current_task = "Processing Green and Red Channel"
        #for track in self.tracks:
        #    track.extract_colored_ROI(red_frames, "red")
        #    track.extract_colored_ROI(green_frames, "green")
        #self.current_task = "idle"

        # global tasks_completed
        # self.current_task = "Processing Green Channel"
        # tasks_completed = 0
        # params["color"] = "green"
        # params["frames"] = green_frames
        # if self.executor is not None:
        #     num_cores = os.cpu_count()
        #     with ProcessPoolExecutor(max_workers=int(num_cores*0.75)) as executor:
        #         self.tracks = list(executor.map(extract_colored_ROI, self.tracks, repeat(params)))
        # else:
        #     self.tracks = list(self.executor.map(extract_colored_ROI, self.tracks, repeat(params)))
        #
        # self.current_task = "Processing Red Channel"
        # tasks_completed = 0
        # params["color"] = "red"
        # params["frames"] = red_channel
        # if self.executor is not None:
        #     with ProcessPoolExecutor(max_workers=int(num_cores*0.75)) as executor:
        #         self.tracks = list(executor.map(extract_colored_ROI, self.tracks, repeat(params)))
        # else:
        #     self.tracks = list(self.executor.map(extract_colored_ROI, self.tracks, repeat(params)))
        #
        # self.current_task = "idle"




    def analyze_all_tracks(self):

        num_cores = os.cpu_count()
        # First fit the displacements by a Gaussian for each tracks



        global tasks_completed

        tasks_completed = 0
        self.current_task = "Fit by Gaussian"
        if self.executor is not None:
            num_cores = os.cpu_count()
            with ProcessPoolExecutor(max_workers=int(num_cores*0.75)) as executor:
                self.tracks = list(executor.map(fit_track_displacement_by_a_gaussian, self.tracks, repeat(params), chunksize=1))
        else:
            self.tracks = list(self.executor.map(fit_track_displacement_by_a_gaussian, self.tracks, repeat(params), chunksize=1))

        tasks_completed = 0
        self.current_task = "Covariance estimator"
        if self.executor is not None:
            num_cores = os.cpu_count()
            with ProcessPoolExecutor(max_workers=int(num_cores*0.75)) as executor:
                self.tracks = list(executor.map(get_D_from_track_covariance, self.tracks, repeat(params)))
        else:
            self.tracks = list(self.executor.map(get_D_from_track_covariance, self.tracks, repeat(params), chunksize=1))

        self.current_task = "idle"



    def exportData(self, filename):
        data = (self.Moyenner)
        np.savetxt(filename, data,newline='\n')












    def get_all_delta_in_chronological_order(self):
        pass

    def compute_global_drift_speed(self, windows_length=None):
        """
        On a besoin d'un tableau ordonée en temps et non par particule.
        :param windows_length:
        :return:
        """
        pass

    def compute_drift_from_centroid(self):
        pass

    def calculate_r_from_full_gauss(self, mode="all"):
        """
        Wa gather ALL the displacement (Δx ANF Δy) from ALL the particle into one single gaussian that is fitted
        :param mode: "all" (default)-> combine x and y, "x" -> only x axis, "y" -> only y axis
        :return:
        """
        list_delta = []
        #FIXME better syntax
        for track in self.tracks:
            if mode == "all":
                list_delta.append(np.diff(track.x))
                list_delta.append(np.diff(track.y))
            elif mode == "x":
                list_delta.append(np.diff(track.x))
            elif mode == "y":
                list_delta.append(np.diff(track.y))

        deltas = np.concatenate(list_delta)
        diff, hist, boundaries, result, center, width = self.gaussian_delta_fit(deltas)
        self.hist_full_gauss = hist
        self.boundaries_full_gauss = hist
        self.result_fit_full_gauss = result


    def generate_brownian_track(self, params_dict):
        #FIXME from params_dict
        T = 293
        eta = 1E-3
        delta_t_ms = 30
        nb_of_frame = 20000
        particle_mean_diam_nm = 50
        particle_diam_sigma_relative = 0
        dim_box_X_micron = 3000
        dim_box_Y_micron = 3000
        dim_box_Z_micron = 50
        nb_particle = 100
        depth_of_focus_micron = 100
        drift_X_microns_per_frame = 0
        drift_Y_microns_per_frame = 0

        self.space_units_nm = 263   # in nm
        self.timeUnits = delta_t_ms/1000   # in s
        self.T = T
        self.eta = eta


        kb = 1.380649E-23

        particle_radiuss = np.random.normal(loc=particle_mean_diam_nm/2, scale=particle_mean_diam_nm/2*particle_diam_sigma_relative, size=nb_particle)
        mean_diff_coeff = kb * T / (6*np.pi*eta*(particle_mean_diam_nm/2*1E-9))
        diff_coeffs = kb * T / (6*np.pi*eta*(particle_radiuss*1E-9))

        # brownian_length = np.sqrt(2*mean_diff_coeff*delta_t_ms*1E-3)
        # mean_nb_spot_pert_track  = 0
        # mean_dwell_time_in_focus_s = (depth_of_focus_micron*1E-9)**2/(2*mean_diff_coeff)

        npParticleType = np.dtype(
            [('x', np.float), ('y', np.float), ('z', np.float), ('Dtx', np.float), ('Dty', np.float),
             ('Dtz', np.float)])
        particles = np.zeros(nb_particle, dtype=npParticleType)


        particles[:]['Dtx'] = particles[:]['Dtz'] = particles[:]['Dty'] = diff_coeffs

        # Initial position -> offset to all position
        r = np.random.rand(nb_particle, 3)
        particles[:]['x'] = r[:, 0] * dim_box_X_micron
        particles[:]['y'] = r[:, 1] * dim_box_Y_micron
        particles[:]['z'] = r[:, 2] * 2*dim_box_Z_micron - dim_box_Z_micron # centered on z = 0 i.e. the focal plane

        #Brownian motion
        # Draw random samples from a normal (Gaussian) distribution.
        dr = np.random.normal(loc=0, scale=1.0, size=(nb_of_frame, nb_particle, 3))

        # Constant drift
        dr[:, :, 0] += drift_X_microns_per_frame
        dr[:, :, 1] += drift_Y_microns_per_frame

        # Construct the brownian trajectory by adding all the displacement
        dr = np.cumsum(dr, axis=0, out=dr)

        # TODO do not create a new array at each iteration
        mvt_evolution = np.zeros((nb_of_frame, nb_particle), dtype=npParticleType)

        # offsetting at t=0 by the initial position
        mvt_evolution[:] = particles

        # Scaling the displacement with the diffusion coefficient
        mvt_evolution[:]['x'] += dr[:, :, 0] * np.sqrt(2 * particles[:]['Dtx'] * self.timeUnits) * 1E6
        mvt_evolution[:]['y'] += dr[:, :, 1] * np.sqrt(2 * particles[:]['Dty'] * self.timeUnits) * 1E6
        mvt_evolution[:]['z'] += dr[:, :, 2] * np.sqrt(2 * particles[:]['Dtz'] * self.timeUnits) * 1E6

        # Extract track from brownian trajectory
        self.tracks = []

        # Une track est un ensemble de spot dont la position est comprise entre :
        # x ∈ [0, dimX] and y ∈ [0, dimY] and z ∈ [-dof, +dof] where dof : depth of focus
        # Les deux premieres conditions sont toujours vérifiées si on applique des conditions aux limites périodiques

        def consecutive(data, stepsize=1):
            # return np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)
            idx = np.r_[0, np.where(np.diff(data) != stepsize)[0] + 1, len(data)]
            return [data[i:j] for i, j in zip(idx, idx[1:])]

        xs = mvt_evolution[:]['x']
        ys = mvt_evolution[:]['y']
        zs = mvt_evolution[:]['z']
        dof = depth_of_focus_micron
        for i in range(nb_particle):
            z = zs[:, i]
            idx_in_focus = np.where(np.logical_and(z < dof, z > -dof))
            if len(idx_in_focus[0]) > 1:  # Test if the list is not empty
                tracks_idx = consecutive(idx_in_focus)
                # FIXME there is a problem with the "consecutive" function that return empty array
                pos_micro_to_pixel = 1/(self.space_units_nm/1000)
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
        self.nTracks = len(self.tracks)
        self.analyze_all_tracks()

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

    # def filter_spots(self, spots, name, value, isabove):
    #     if isabove:
    #         spots = spots[spots[name] > value]
    #     else:
    #         spots = spots[spots[name] < value]
    #
    #     return spots


    def loadxmlTrajs(self, trackmate_xml_path):
        """
        Load xml files into a python dictionary with the following structure:
            tracks = {'0': {'nSpots': 20, 'trackData': numpy.array(t, x, y, z) }}
        Tracks should be xml file from 'Export tracks to XML file',
        that contains only track info but not the features.
        Similar to what 'importTrackMateTracks.m' needs.
        """
        # TODO multicore and loading bar in the GUI
        try:
            tree = et.parse(trackmate_xml_path)
        except OSError:
            print('Failed to read XML file {}.'.format(xlmfile))
        root = tree.getroot()
        # print(root.attrib)  # or extract metadata
        self.nTracks = int(root.attrib['nTracks'])
        self.frameInterval = float(root.attrib['frameInterval'])
        self.space_units_nm = root.attrib['spaceUnits']
        self.timeUnits = root.attrib['timeUnits']

        # Working with numpy array -> TOO SLOW
        # self.tracks = []
        # for i in range(self.nTracks):
        #     track = Track()
        #     track.nSpots = int(root[i].attrib['nSpots'])
        #     # FIXME more pythonic expression
        #     track.t, track.x, track.y, track.z = np.zeros(track.nSpots), np.zeros(track.nSpots), np.zeros(track.nSpots), np.zeros(track.nSpots)
        #     # self.tracks[trackIdx]['nSpots'] = nSpots
        #     # trackData = np.array([]).reshape(0, 4)
        #     for j in range(track.nSpots):
        #         track.t[j] = float(root[i][j].attrib['t'])
        #         track.x[j] = float(root[i][j].attrib['x'])
        #         track.y[j] = float(root[i][j].attrib['y'])
        #         track.z[j] = float(root[i][j].attrib['z'])


        # self.tracks = [Track()]*self.nTracks
        self.tracks = [Track() for k in range(self.nTracks)]
        for i in range(self.nTracks):
            trackIdx = str(i)
            track = self.tracks[i]
            track.t = np.zeros(track.nSpots)
            track.x = np.zeros(track.nSpots)
            track.y = np.zeros(track.nSpots)
            track.z = np.zeros(track.nSpots)
            # self.tracks_dict[trackIdx] = {}
            nSpots = int(root[i].attrib['nSpots'])
            # self.tracks_dict[trackIdx]['nSpots'] = nSpots
            trackData = np.array([]).reshape(0, 4)
            for j in range(nSpots):
                t = float(root[i][j].attrib['t'])
                x = float(root[i][j].attrib['x'])
                y = float(root[i][j].attrib['y'])
                z = float(root[i][j].attrib['z'])
                spotData = np.array([t, x, y, z])
                trackData = np.vstack((trackData, spotData))
            # self.tracks_dict[trackIdx]['trackData'] = trackData

            track.nSpots = nSpots
            track.t = trackData[:, 0]
            track.x = trackData[:, 1]
            track.y = trackData[:, 2]
            track.z = trackData[:, 3]



        # Analyze track
        # MonoCore implementation
        i = 0
        for track in self.tracks :
            # print(i)
            # track.gaussian_delta_fit()
            # track.msd()
            # track.covariance()
            i += 1

        # # Muti core implementation
        # nb_of_workers = 4
        # p = mp.Pool(nb_of_workers)
        #
        # def analyze_track(track_num):
        #     print(track_num)
        #     self.tracks[i].gaussian_delta_fit()
        #     self.tracks[i].msd()
        #     self.tracks[i].covariance()
        #     return track_num
        #
        #
        # results = [p.apply_async(analyze_track, args=(i, )) for i in range(self.nTracks)]
        # # track.gaussian_delta_fit()
        # # track.msd()
        # # track.covariance()
        # # print(result)
        #
        # # Gs = [p.get() for p in results]
        # # print(Gs)
        #
        # Create statistics from  Track analyze :
        # TODO
        # Calculate Dx, Dy, rx, ry, etc


    def trackmate_peak_import(self, trackmate_xml_path, get_tracks=False):
        """
        From pyTrackmate

        Import detected peaks with TrackMate Fiji plugin.

        Parameters
        ----------
        trackmate_xml_path : str
            TrackMate XML file path.
        get_tracks : boolean
            Add tracks to label
        """

        root = et.fromstring(open(trackmate_xml_path).read())

        objects = []
        object_labels = {'FRAME': 't_stamp',
                         'POSITION_T': 't',
                         'POSITION_X': 'x',
                         'POSITION_Y': 'y',
                         'POSITION_Z': 'z',
                         'MEAN_INTENSITY': 'I',
                         'ESTIMATED_DIAMETER': 'w',
                         'QUALITY': 'q',
                         'ID': 'spot_id',
                         'MEAN_INTENSITY': 'mean_intensity',
                         'MEDIAN_INTENSITY': 'median_intensity',
                         'MIN_INTENSITY': 'min_intensity',
                         'MAX_INTENSITY': 'max_intensity',
                         'TOTAL_INTENSITY': 'total_intensity',
                         'STANDARD_DEVIATION': 'std_intensity',
                         'CONTRAST': 'contrast',
                         'SNR': 'snr'}

        features = root.find('Model').find('FeatureDeclarations').find('SpotFeatures')
        features = [c.get('feature') for c in list(features)] + ['ID']

        spots = root.find('Model').find('AllSpots')
        trajs = pd.DataFrame([])
        objects = []
        for frame in spots.findall('SpotsInFrame'):
            for spot in frame.findall('Spot'):
                single_object = []
                for label in features:
                    single_object.append(spot.get(label))
                objects.append(single_object)

        trajs = pd.DataFrame(objects, columns=features)
        trajs = trajs.astype(np.float)

        # Apply initial filtering
        initial_filter = root.find("Settings").find("InitialSpotFilter")

        trajs = self.filter_spots(trajs,
                             name=initial_filter.get('feature'),
                             value=float(initial_filter.get('value')),
                             isabove=True if initial_filter.get('isabove') == 'true' else False)

        # Apply filters
        spot_filters = root.find("Settings").find("SpotFilterCollection")

        for spot_filter in spot_filters.findall('Filter'):

            trajs = self.filter_spots(trajs,
                                 name=spot_filter.get('feature'),
                                 value=float(spot_filter.get('value')),
                                 isabove=True if spot_filter.get('isabove') == 'true' else False)

        trajs = trajs.loc[:, object_labels.keys()]
        trajs.columns = [object_labels[k] for k in object_labels.keys()]
        trajs['label'] = np.arange(trajs.shape[0])

        # Get tracks
        if get_tracks:
            filtered_track_ids = [int(track.get('TRACK_ID')) for track in root.find('Model').find('FilteredTracks').findall('TrackID')]

            label_id = 0
            trajs['label'] = np.nan

            tracks = root.find('Model').find('AllTracks')
            for track in tracks.findall('Track'):

                track_id = int(track.get("TRACK_ID"))
                if track_id in filtered_track_ids:

                    spot_ids = [(edge.get('SPOT_SOURCE_ID'), edge.get('SPOT_TARGET_ID'), edge.get('EDGE_TIME')) for edge in track.findall('Edge')]
                    spot_ids = np.array(spot_ids).astype('float')[:, :2]
                    spot_ids = set(spot_ids.flatten())

                    trajs.loc[trajs["spot_id"].isin(spot_ids), "label"] = label_id
                    label_id += 1

            # Label remaining columns
            single_track = trajs.loc[trajs["label"].isnull()]
            trajs.loc[trajs["label"].isnull(), "label"] = label_id + np.arange(0, len(single_track))


        print(trajs)
        return trajs

if __name__ == "__main__":
    core = AnalyzeTrackCore()
    print(core.space_units_nm)
    core.generate_brownian_track(params_dict=None)
    pass


