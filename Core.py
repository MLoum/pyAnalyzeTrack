import matplotlib.pyplot as plt
import numpy as np
import xml.etree.cElementTree as et
from lmfit import Model

import pandas as pd
import multiprocessing as mp


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
        self.t, self.x, self.y, self.z = None, None, None, None
        self.diff_x, self.diff_y = None, None

        self.drift_x, self.drift_y = None, None
        self.Dx_gauss, self.Dy_gauss = None, None
        self.Dx_msd, self.Dy_msd = None, None
        self.Dx_cov, self.Dy_cov = None, None
        self.result_gauss_x, self.result_gauss_y = None, None

        self.r_gauss, self.r_cov = 0, 0
        self.error_r_gauss, self.error_r_cov = 0, 0



        self.isFiltered = False

    def gaussian_delta_fit(self):
        """
        Cf plutôt la même méthode dans la classe AnalyzeTrackCore
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

        x_diff = np.diff(self.x)
        hist_x, boundaries_x = np.histogram(x_diff)
        boundaries_x = boundaries_x[0:-1]
        gmodel = Model(gaussian)
        params = gmodel.make_params(cen=np.mean(x_diff), amp=np.max(hist_x), wid=np.std(x_diff))

        result_x = gmodel.fit(hist_x, params, x=boundaries_x)
        self.result_gauss_x = result_x
        self.drift_x = result_x.params["cen"]
        self.width_x = result_x.params["wid"]

        y_diff = np.diff(self.y)
        hist_y, boundaries_y = np.histogram(y_diff)
        boundaries_y = boundaries_y[0:-1]
        gmodel = Model(gaussian)
        params = gmodel.make_params(cen=np.mean(y_diff), amp=np.max(hist_y), wid=np.std(y_diff))
        result_y = gmodel.fit(hist_y, params, x=boundaries_y)
        self.result_gauss_y = result_y
        self.drift_y = result_y.params["cen"]
        self.width_y = result_x.params["wid"]

        return hist_x, boundaries_x, hist_y, boundaries_y, result_x, result_y

    def msd(self):
        """
        A implementer par principe même si la méthode n'est pas la bonne pour le mouvement brownien libre.
        :return:
        """
        pass




    def test_free_diffusion_via_MSD(self, algo="None"):
        """
        Pour un mouvement brownien libre :
        <Δx_n²> = 2 D Δt  + 2 (σ² - 2 D R Δt)
        <Δx_n Δx_n+1> ≠ 0 = - (σ² - 2 D R Δt)
        Tous les autres <Δx_n Δx_m> = 0

        On teste ici si les <Δx_n Δx_m> = 0. Ils doivent être reparties autour de zero.
        Cela est très facilement remis en question s'il y a du drift qui n'est pas bien corrigé.
        :param algo:
        :return:
        """
        nb_msd = min(15, self.nSpots)
        self.test_msd_x = np.zeros(nb_msd)
        self.test_msd_y = np.zeros(nb_msd)

        for i in range(nb_msd):
            self.test_msd_x[i] = np.mean(self.diff(self.x, i+1), algo)
            self.test_msd_y[i] = np.mean(self.diff(self.y, i + 1), algo)

    def test_free_diffusion_via_periodogram(self):
        """
        Analyse du spectre en Discrete Sinus Fourier Transform
        :return:
        """
        # TODO
        pass

    def diff(self, array, n=1, algo="None"):
        """

        :param array:
        :param n:
        :return:
        """
        if algo == "None":
            return array[n:] - array[:-n]
        elif algo == "self":
            (array[n:] - np.mean(array[n:])) - (array[:-n] - np.mean(array[:-n]))



    def evol_drift(self, window_length_ratio):
        """
        Etude de la moyenne des Delta x (<Δx>) au cours de la trajectoire.
        :param window_length_ratio:
        :return:
        """
        def running_mean(x, N):
            cumsum = np.cumsum(np.insert(x, 0, 0))
            return (cumsum[N:] - cumsum[:-N]) / float(N)

        filter_length = int(self.x * window_length_ratio)
        return running_mean(self.x, filter_length)

    def get_deltas(self, algo="None"):
        """
        Return the list of all displacement Δx, defined as Δx = x[i+1] - x[i], while taking into account an eventual drift (cf algo)
        :param algo: Algorithm for drift compensation.
        "self" remove the mean of the data from all displacements
        :return:
        """
        if algo == "None":
            return np.diff(self.x), np.diff(self.y)
        elif algo == "self":
            return np.diff(self.x) - np.mean(self.x), np.diff(self.y) - np.mean(self.y)
        elif algo == "neighbors":
            # TODO
            return np.diff(self.x), np.diff(self.y)

class AnalyzeTrackCore():
    """

    """
    def __init__(self):
        self.tracks = None
        self.nTracks = None
        self.frameInterval = None
        self.spaceUnits = 263   # in nm
        self.timeUnits = 1/30   # in s
        self.T = 293    #in K
        self.eta = 0.001    # in Pa.s
        self.sigma_already_known = False

        self.algo_drift = "self"


    def viscosity_from_temperature(self, solvent="water", T=293):
        # TODO tabuler la viscosité de l'eau.
        if solvent == "water":
            return 1E-3

        return None


    def covariance(self, Delta_t=1/60, R=1/6, sigma_square=None, algo_drift="self"):
        """
        Calcul du coefficeint de diffusion D à partir de la covariance :

        Vestergaard, C. L., Blainey, P. C., & Flyvbjerg, H. (2014).
        Optimal estimation of diffusion coefficients from single-particle trajectories.
        Physical Review E - Statistical, Nonlinear, and Soft Matter Physics, 89(2).
        https://doi.org/10.1103/PhysRevE.89.022726
        :param Delta_t:
        :param R:
        :param sigma_square: Erreur de localisation qui peut être mesurée ou estimée à partir des
        :return:
        """
        self.Moyennex = float(0)
        self.i = 0
        # TODO sigma (bruit localisation) à partir des données de la track ou alors sigma
        # moyen sur l'ensemble des tracks ou alors sigma utilisateur

        # TODO work in progress
        for track in self.tracks:
            if track.nSpots < 10 :
                track.isFiltered = True
            if track.isFiltered is True :
                track.r_cov = np.nan
            else:
                self.i=self.i+1

                diff_x_1 = self.calculate_displacement(track.x*265*10**-9, algo="None")
                diff_x_2 = self.calculate_displacement(track.x*265*10**-9, algo="covar")
                drift = np.nanmean(diff_x_1)
                MSD = np.mean((diff_x_1-drift) ** 2)
                covariance = np.mean((diff_x_1[:-1]-drift) * (diff_x_2-2*drift))

                if sigma_square is None:
                    track.Dx = MSD / (2*Delta_t) + covariance/Delta_t
                    sigma_square = R*MSD + (2*R-1)*covariance
                    r_cov_1 = (1.38*10**(-23)*293)/(6*np.pi*0.001*track.Dx)
                    track.r_cov = round(r_cov_1,12)
                else:
                    sigma_square = R * MSD + (2 * R - 1) * covariance
                    track.D_x = MSD - 2*sigma_square / (2*(1-2*R)*Delta_t)
                    r_cov_1 =(1.38*10**(-23)*293)/(6*np.pi*0.001*track.D_x)
                    track.r_cov = round(r_cov_1, 12)
            if track.r_cov == np.nan:
                self.Moyennex = self.Moyennex
            else:
                self.Moyennex = self.Moyennex + r_cov_1 / self.i
        #TODO check and y axis
        print(self.Moyennex)
        #TODO variance on D


    def load_txt(self, filepath):

        # TODO verifier que la position de l'export est toujours la même. Sinon, coder un truc
        # On skip la premiere colone qui n'est pas un nombre.
        raw = np.loadtxt(filepath, skiprows=1, usecols=np.arange(1, 21))

        #Label ID TRACK_ID QUALITY POSITION_X POSITION_Y POSITION_Z POSITION_T FRAME RADIUS VISIBILITY MANUAL_COLOR MEAN_INTENSITY MEDIAN_INTENSITY MIN_INTENSITY MAX_INTENSITY TOTAL_INTENSITY STANDARD_DEVIATION ESTIMATED_DIAMETER CONTRAST SNR
        nb_lines = np.shape(raw)[0]
        self.nTracks = int(raw[-1, 1])  #Track_ID

        self.tracks = [Track() for k in range(self.nTracks)]

        unique, unique_index = np.unique(raw[:,1], return_index=True)
        i = 0
        for i in range(self.nTracks):
            data = raw[unique_index[i]:unique_index[i+1], :]
            self.tracks[i].quality = data[:, 2]
            self.tracks[i].x = data[:, 3]
            self.tracks[i].y = data[:, 4]
            self.tracks[i].z = data[:, 5]
            self.tracks[i].t = data[:, 6]
            self.tracks[i].diff_x = np.diff(self.tracks[i].x,1)
            self.tracks[i].diff_y = np.diff(self.tracks[i].y,1)
            # TODO le reste
            self.tracks[i].nSpots = self.tracks[i].x.size
            i += 1

        # raw = np.transpose(raw)
        # Chronological order :
        # Serach for all lines taht shares the same time t
        test = raw[:, 6]
        unique, unique_index, unique_inverse, unique_counts = np.unique(raw[:, 6], return_index=True, return_inverse= True, return_counts=True)
        self.nb_track_per_frame = unique_counts
        self.id_of_track_at_time_t = None

        #TODO multicore
        self.analyze_all_tracks()

    def analyze_all_tracks(self):
        self.calculate_r_from_gauss()
        self.covariance()

    def calculate_displacement(self, pos, algo="None", step=1):
        """
        Return the list of all displacement Δx, defined as Δx = x[i+1] - x[i], while taking into account an eventual drift (cf algo)
        :param algo: Algorithm for drift compensation.
        "self" remove the mean of the data from all displacements
        :return:
        """
        if algo == "None":
            return np.diff(pos,step)
        elif algo == "self":
            return np.diff(pos,step) - np.mean(pos)
        elif algo == "neighbors":
            # TODO
            return np.diff(pos), np.diff(pos)
        elif algo == "covar":
            return np.diff(pos,step+1)


    def gaussian_delta_fit(self, pos, algo_drift_compensation="None"):
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
        hist, boundaries = np.histogram(diff)
        boundaries = boundaries[0:-1]
        gmodel = Model(gaussian, nan_policy='raise')
        max_, min_ = np.max(hist), np.min(hist)
        max_bound, min_bound = np.max(boundaries), np.min(boundaries)
        params = gmodel.make_params(cen=np.mean(diff), amp=max_, wid=np.std(diff))
        params["amp"].min = 2 * max_
        params["wid"].min = 0
        params["cen"].min = min_bound

        params["cen"].max = max_bound
        params["wid"].max = max_bound - min_bound

        try :
            result = gmodel.fit(hist, params, x=boundaries)
        except ValueError:
            return None, None, None, None, None, None

        return diff, hist, boundaries, result, result.params["cen"], result.params["wid"]


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


    def calculate_r_from_gauss(self):
        """
        Var (Δx) = <Δx²> - (<Δx>)²
        <Δx²> est la MSD est on peut en déduire le coefficient de diffusion via  <Δx²> = 2 D Δt
        (<Δx>)² est le carré de la moyenne, on peut en déduire une estimation, c'est le centre de la gaussienne.
        Var (Δx) provient de l'ajustement de l'histogramme par une gaussienne.
        :return:
        """
        kb_ = 1.38E-23
        i = 0
        for track in self.tracks:
            if i==963:
                track.r_gauss = -1
                dummy = 1
                i += 1
                continue
            print(i)
            i += 1
            if track.isFiltered:
                track.r_gauss = -1
                continue
            if track.nSpots < 10:
                track.r_gauss = -1
                continue

            diff_x, hist_x, boundaries_x, result_x, center_x, width_x = self.gaussian_delta_fit(track.x, self.algo_drift)
            # width <->  2 σ² where σ is the standart deviation and σ² the variance
            # FIXME
            if diff_x is None:
                continue
            #FIXME passer de pixel à nm
            msd = width_x.value*2 + center_x.value**2
            msd *= self.spaceUnits * 1E-9   # 1E-9 nm -> m
            track.Dx_gauss = msd/(2*self.timeUnits)
            track.rx_gauss = kb_ * self.T / (6 * np.pi * self.eta * track.Dx_gauss)

            diff_y, hist_y, boundaries_y, result_y, center_y, width_y = self.gaussian_delta_fit(track.y, self.algo_drift)
            if diff_y is None:
                continue
            msd = width_y.value/2 + center_y**2
            msd *= self.spaceUnits * 1E-9
            track.Dy_gauss = msd/(2*self.timeUnits)
            track.ry_gauss = kb_ * self.T / (6 * np.pi * self.eta * track.Dy_gauss)

            #FIXME mean or np.sqrt(track.rx_gauss**2 + track.ry_gauss**2)/2
            track.r_gauss = (track.rx_gauss + track.ry_gauss)/2


    def generate_free_brownian_motion(self, params):
        #FIXME get info from params
        self.nb_part = 100
        for n in self.nb_part:
            # r = np.
            pass
        pass

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
            tree = et.parse(trackmate_xml_path);
        except OSError:
            print('Failed to read XML file {}.'.format(xlmfile))
        root = tree.getroot()
        # print(root.attrib)  # or extract metadata
        self.nTracks = int(root.attrib['nTracks'])
        self.frameInterval = float(root.attrib['frameInterval'])
        self.spaceUnits = root.attrib['spaceUnits']
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

