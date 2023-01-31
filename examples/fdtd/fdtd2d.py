from .grid import Grid
from .boundaries import PML
from .sources import PointSource

from numba import njit,prange
from numpy import float32

# TM
class FDTD2d(Grid):
    def __init__(self,media,constants,ddx,f,DTYPE=float32):
        super().__init__(media,constants,ddx,f,DTYPE)
        self.set_boundaries('None')
        self.set_sources('PointSource')


    def set_boundaries(self,boundaries):
        self.boundaries = boundaries
        if boundaries == 'PML':
            self.boundary = PML(self.media.shape,self.DTYPE)


    def set_sources(self,sources):
        self.sources = sources
        if self.sources == 'PointSource':
            self.source = PointSource(self.f,self.dt)


    @staticmethod
    @njit(parallel=True)
    def calculate_dz_field(dz,hx,hy):
        x,y = dz.shape
        for i in prange(1,x):
            for j in prange(1,y):
                dz[i, j] = dz[i, j] + 0.5 * (hy[i,j] - hy[i-1,j] - hx[i,j] + hx[i,j-1])

        return dz


    @staticmethod
    @njit(parallel=True)
    def calculate_dz_field_pml(dz,hx,hy,pdx2,pdx3,pdy2,pdy3):
        x,y = dz.shape
        for i in prange(1,x):
            for j in prange(1,y):
                dz[i, j] = pdx3[i] * pdy3[j] * dz[i, j] \
                    + pdx2[i] * pdy2[j] * 0.5 * (hy[i, j] - hy[i - 1, j] - hx[i, j] + hx[i, j - 1])

        return dz


    @staticmethod
    @njit(parallel=True)
    def calculate_ez_field(ez,iz,dz,cez,cezl,media):
        x,y = ez.shape
        for i in prange(1, x):
            for j in prange(1, y):
                idx = media[i,j]
                ez[i,j] = cez[idx] * (dz[i,j] - iz[i,j])
                iz[i,j] = iz[i,j] + cezl[idx] * ez[i,j]

        return ez,iz


    @staticmethod
    @njit(parallel=True)
    def calculate_hx_field(hx,ez):
        x,y = hx.shape
        for i in prange(x-1):
            for j in prange(y-1):
                hx[i,j] = hx[i,j] + 0.5 * (ez[i,j] - ez[i,j+1])

        return hx


    @staticmethod
    @njit(parallel=True)
    def calculate_hy_field(hy,ez):
        x,y = hy.shape
        for i in prange(x-1):
            for j in prange(y-1):
                hy[i,j] = hy[i,j] + 0.5 * (ez[i+1,j] - ez[i,j])

        return hy


    @staticmethod
    @njit(parallel=True)
    def calculate_hx_field_pml(hx,ez,ihx,phx1,phy2,phy3):
        x,y = hx.shape
        for i in prange(x-1):
            for j in prange(y-1):
                ihx[i, j] = ihx[i, j] + (ez[i, j] - ez[i, j + 1])
                hx[i, j] = phy3[j] * hx[i, j] + phy2[j] * (0.5 * (ez[i, j] - ez[i, j + 1]) + phx1[i] * ihx[i, j])

        return hx,ihx


    @staticmethod
    @njit(parallel=True)
    def calculate_hy_field_pml(hy,ez,ihy,phx2,phx3,phy1):
        x,y = hy.shape
        for i in prange(x-1):
            for j in prange(y-1):
                ihy[i, j] = ihy[i, j] + (ez[i, j] - ez[i + 1, j])
                hy[i, j] = phx3[i] * hy[i, j] - phx2[i] * (0.5 * (ez[i, j] - ez[i + 1, j]) + phy1[j] * ihy[i, j])

        return hy,ihy


    def update_d_fields(self):

        if self.boundaries == 'None':
            self.dz = self.calculate_dz_field(self.dz,self.hx,self.hy)

        elif self.boundaries == 'PML':
            self.dz = self.calculate_dz_field_pml(self.dz,self.hx,self.hy,self.boundary.pdx2,self.boundary.pdx3,self.boundary.pdy2,self.boundary.pdy3)


    def update_e_fields(self):

        self.ez,self.iz = self.calculate_ez_field(self.ez,self.iz,self.dz,self.cez,self.cezl,self.media)


    def update_h_fields(self):

        if self.boundaries == 'None':
            self.hx = self.calculate_hx_field(self.hx,self.ez)
            self.hy = self.calculate_hy_field(self.hy,self.ez)

        elif self.boundaries == 'PML':
            self.hx,self.boundary.ihx = self.calculate_hx_field_pml(self.hx,self.ez,self.boundary.ihx,self.boundary.phx1,self.boundary.phy2,self.boundary.phy3)
            self.hy,self.boundary.ihy = self.calculate_hy_field_pml(self.hy,self.ez,self.boundary.ihy,self.boundary.phx2,self.boundary.phx3,self.boundary.phy1)


    def run(self):
        self.update_d_fields()
        self.dz[self.nxc,self.nyc] = self.source.update_source()
        self.update_e_fields()
        self.dz[self.nxc,self.nyc] = self.source.update_source()
        self.update_h_fields()


    def run_animation(self,nsteps):
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        import matplotlib.ticker as ticker
        import matplotlib.colors as mcolors
        from numpy.ma import masked_where

        colors = [(0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 1, 0)] # blue, green, red, yellow
        custom_map = mcolors.ListedColormap(colors)

        ims = []
        fig,ax = plt.subplots()
        ax.set_facecolor([0.0,0.0,0.5])
        for _ in range(nsteps):
            im1 = ax.imshow(abs(self.ez.T), cmap='jet', origin='lower')
            im2 = ax.imshow(masked_where(self.media.T == 0, self.media.T),interpolation='nearest',cmap=custom_map,alpha=0.5)
            ims.append([im1,im2])
            self.run()
        cbar = fig.colorbar(im1)
        cbar.ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        plt.close()
        plt.rcParams['animation.embed_limit'] = 50
        ani = animation.ArtistAnimation(fig, ims, interval=50, blit=False)

        return ani

