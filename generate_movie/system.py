class Camera:
    def __init__(self, size, axial_res, lateral_res, magnification, offset_x, offset_y, encoding_precision):
        self.size = size                # number of pixels of the camera
        self.axial_res = axial_res      # used to precalculate PSF in the z direction (µm)
        self.lateral_res = lateral_res  # real pixel size (µm)
        self.M = magnification
        self.offset_x = offset_x        # Only used to calculate the intensity of lightsheet with some offset
        self.offset_y = offset_y        # Only used to calculate the intensity of lightsheet with some offset
        self.encoding_precision = encoding_precision  # number of bits used to encode color

    @property
    def pixel_to_size(self):
        return self.lateral_res / self.M  # in µm

    def size_to_pixel(self, size):
        return size * self.M / self.lateral_res  # in pixels


class Microscope:
    def __init__(self, wavelength, M, NA, ng0, ng, ni0, ni, ns, ti0, tg0, tg, zd0, covertop_z):
        self.wavelength = wavelength    # microns
        self.M = M                      # Magnification
        self.NA = NA                    # numerical aperture
        self.ng0 = ng0                  # coverslip RI design value
        self.ng = ng                    # coverslip RI experimental value
        self.ni0 = ni0                  # immersion medium RI design value
        self.ni = ni                    # immersion medium RI experimental value
        self.ns = ns                    # specimen refractive index (RI)
        self.ti0 = ti0                  # microns, working distance (immersion medium thickness) design value
        self.tg0 = tg0                  # microns, coverslip thickness design value
        self.tg = tg                    # microns, coverslip thickness experimental value
        self.zd0 = zd0                  # microscope tube length (in microns).
        self.covertop_z = covertop_z    # [µm] focus just 'above' interface (inside the sample medium)

    @property
    def parameters(self):
        mp = {"M": self.M,
              "NA": self.NA,
              "ng0": self.ng0,
              "ng": self.ng,
              "ni0": self.ni0,
              "ni": self.ni,
              "ns": self.ns,
              "ti0": self.ti0,
              "tg": self.tg0,
              "tg0": self.tg,
              "zd0": self.zd0}
        return mp
