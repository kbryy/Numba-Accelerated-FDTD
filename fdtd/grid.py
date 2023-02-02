import numpy as np
from numba import njit
from . import constants as const

class Grid:

    def __init__(self,media,constants,ddx,f,DTYPE):

        self.DTYPE     = DTYPE
        self.constants = np.array(constants)
        self.media     = np.array(media,dtype=np.int8)
        self.idn       = int(np.max(media) + 2)  # id:-1 完全導体 id:0〜任意媒体
        self.f         = f
        self.ddx       = ddx
        self.dt        = ddx / (2.0 * const.C)

        if media.ndim == 1:

            self.nx   = media.shape[0]
            self.nxc  = self.nx // 2
            # self.dx   = np.zeros(self.nx,dtype=self.DTYPE)
            self.hy   = np.zeros(self.nx,dtype=self.DTYPE)
            self.ex   = np.zeros(self.nx,dtype=self.DTYPE)
            self.ix   = np.zeros(self.nx,dtype=self.DTYPE)
            self.cex  = np.zeros(self.idn,dtype=self.DTYPE)
            self.cexl = np.zeros(self.idn,dtype=self.DTYPE)
            self.calculate_media_coefficients_1d()


        elif media.ndim == 2:

            self.nx   = media.shape[0]
            self.ny   = media.shape[1]
            self.nxc  = self.nx // 2
            self.nyc  = self.ny // 2
            self.ez   = np.zeros((self.nx,self.ny),dtype=self.DTYPE)
            self.iz   = np.zeros((self.nx,self.ny),dtype=self.DTYPE)
            self.hx   = np.zeros((self.nx,self.ny),dtype=self.DTYPE)
            self.hy   = np.zeros((self.nx,self.ny),dtype=self.DTYPE)
            self.cez  = np.zeros(self.idn,dtype=self.DTYPE)
            self.cezl = np.zeros(self.idn,dtype=self.DTYPE)
            self.calculate_media_coefficients_2d()


        elif media.ndim == 3:

            self.nx   = media.shape[0]
            self.ny   = media.shape[1]
            self.nz   = media.shape[2]
            self.nxc  = self.nx // 2
            self.nyc  = self.ny // 2
            self.nzc  = self.nz // 2
            self.ex   = np.zeros((self.nx,self.ny,self.nz),dtype=self.DTYPE)
            self.ey   = np.zeros((self.nx,self.ny,self.nz),dtype=self.DTYPE)
            self.ez   = np.zeros((self.nx,self.ny,self.nz),dtype=self.DTYPE)
            self.ix   = np.zeros((self.nx,self.ny,self.nz),dtype=self.DTYPE)
            self.iy   = np.zeros((self.nx,self.ny,self.nz),dtype=self.DTYPE)
            self.iz   = np.zeros((self.nx,self.ny,self.nz),dtype=self.DTYPE)
            self.hx   = np.zeros((self.nx,self.ny,self.nz),dtype=self.DTYPE)
            self.hy   = np.zeros((self.nx,self.ny,self.nz),dtype=self.DTYPE)
            self.hz   = np.zeros((self.nx,self.ny,self.nz),dtype=self.DTYPE)
            self.cex  = np.zeros(self.idn,dtype=self.DTYPE)
            self.cey  = np.zeros(self.idn,dtype=self.DTYPE)
            self.cez  = np.zeros(self.idn,dtype=self.DTYPE)
            self.cexl = np.zeros(self.idn,dtype=self.DTYPE)
            self.ceyl = np.zeros(self.idn,dtype=self.DTYPE)
            self.cezl = np.zeros(self.idn,dtype=self.DTYPE)
            self.calculate_media_coefficients_3d()


    def calculate_media_coefficients_1d(self):
        dt    = self.dt
        epsz  = const.EPS0
        for i in range(self.idn-1):
            epsr  = self.constants[i,0]
            sigma = self.constants[i,1]
            self.cex[i]  = 1 / (epsr + (sigma * dt / epsz))
            self.cexl[i] = sigma * dt / epsz

    def calculate_media_coefficients_2d(self):
        dt    = self.dt
        epsz  = const.EPS0
        for i in range(self.idn-1):
            epsr  = self.constants[i,0]
            sigma = self.constants[i,1]
            self.cez[i]  = 1 / (epsr + (sigma * dt / epsz))
            self.cezl[i] = sigma * dt / epsz


    def calculate_media_coefficients_3d(self):
        dt    = self.dt
        epsz  = const.EPS0
        for i in range(self.idn-1):
            epsr  = self.constants[i,0]
            sigma = self.constants[i,1]
            self.cex[i]  = 1 / (epsr + (sigma * dt / epsz))
            self.cey[i]  = 1 / (epsr + (sigma * dt / epsz))
            self.cez[i]  = 1 / (epsr + (sigma * dt / epsz))
            self.cexl[i] = sigma * dt / epsz
            self.ceyl[i] = sigma * dt / epsz
            self.cezl[i] = sigma * dt / epsz

    @staticmethod
    @njit
    def calculate_amp(x,y,z):
        return np.sqrt((x**2 + y**2 + z**2))


