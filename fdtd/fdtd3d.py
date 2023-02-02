from .grid import Grid
from .boundaries import PML
from .sources import PointSource

from numba import njit,prange
from numpy import float32

# TM
class FDTD3d(Grid):
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
    def calculate_ex_field(ex,hy,hz,ix,cex,cexl,media):
        x,y,z = ex.shape
        for i in prange(1, x):
            for j in prange(1, y):
                for k in prange(1, z):
                    curl_e = ex[i, j, k] + 0.5 * (hz[i, j, k] - hz[i, j - 1, k] - hy[i, j, k] + hy[i, j, k - 1])
                    idx = media[i,j,k]
                    ex[i, j, k] = cex[idx] * (curl_e - ix[i, j, k])
                    ix[i, j, k] = ix[i, j, k] + cexl[idx] * ex[i, j, k]
        return ex, ix


    @staticmethod
    @njit(parallel=True)
    def calculate_ey_field(ey,hx,hz,iy,cey,ceyl,media):
        x,y,z = ey.shape
        for i in prange(1, x):
            for j in prange(1, y):
                for k in prange(1, z):
                    curl_e      = ey[i, j, k] + 0.5 * (hx[i, j, k] - hx[i, j, k - 1] - hz[i, j, k] + hz[i - 1, j, k])
                    idx         = media[i,j,k]
                    ey[i, j, k] = cey[idx] * (curl_e - iy[i, j, k])
                    iy[i, j, k] = iy[i, j, k] + ceyl[idx] * ey[i, j, k]
        return ey, iy


    @staticmethod
    @njit(parallel=True)
    def calculate_ez_field(ez,hx,hy,iz,cez,cezl,media):
        x,y,z = ez.shape
        for i in prange(1, x):
            for j in prange(1, y):
                for k in prange(1, z):
                    curl_e = ez[i, j, k] + 0.5 * (hy[i, j, k] - hy[i - 1, j, k] - hx[i, j, k] + hx[i, j - 1, k])
                    idx = media[i,j,k]
                    ez[i, j, k] = cez[idx] * (curl_e - iz[i, j, k])
                    iz[i, j, k] = iz[i, j, k] + cezl[idx] * ez[i, j, k]
        return ez, iz


    @staticmethod
    @njit(parallel=True)
    def calculate_ex_field_pml(ex,hy,hz,ix,cex,cexl,media,iex,pdx1,pdy2,pdy3,pdz2,pdz3):
        x,y,z = ex.shape
        for i in prange(1, x):
            for j in prange(1, y):
                for k in prange(1, z):
                    iex[i, j, k] = iex[i, j, k] + (hz[i, j, k] - hz[i, j - 1, k] - hy[i, j, k] + hy[i, j, k - 1])
                    curl_e = pdy3[j] * pdz3[k] * ex[i, j, k] \
                                + pdy2[j] * pdz2[k] * (0.5 * (hz[i, j, k] - hz[i, j - 1, k] - hy[i, j, k] + hy[i, j, k - 1]) + pdx1[i] * iex[i, j, k])
                    idx = media[i,j,k]
                    ex[i, j, k]  = cex[idx] * (curl_e - ix[i, j, k])
                    ix[i, j, k]  = ix[i, j, k] + cexl[idx] * ex[i, j, k]
        return ex,ix,iex


    @staticmethod
    @njit(parallel=True)
    def calculate_ey_field_pml(ey,hx,hz,iy,cey,ceyl,media,iey,pdx2,pdx3,pdy1,pdz2,pdz3):
        x,y,z = ey.shape
        for i in prange(1, x):
            for j in prange(1, y):
                for k in prange(1, z):
                    iey[i, j, k] = iey[i, j, k] + (hx[i, j, k] - hx[i, j, k - 1] - hz[i, j, k] + hz[i - 1, j, k])
                    curl_e = pdx3[i] * pdz3[k] * ey[i, j, k] \
                        + pdx2[i] * pdz2[k] * (0.5 * (hx[i, j, k] - hx[i, j, k - 1] - hz[i, j, k] + hz[i - 1, j, k]) + pdy1[j] * iey[i, j, k])
                    idx = media[i,j,k]
                    ey[i, j, k] = cey[idx] * (curl_e - iy[i, j, k])
                    iy[i, j, k] = iy[i, j, k] + ceyl[idx] * ey[i, j, k]
        return ey, iy, iey


    @staticmethod
    @njit(parallel=True)
    def calculate_ez_field_pml(ez,hx,hy,iz,cez,cezl,media,iez,pdx2,pdx3,pdy2,pdy3,pdz1):
        x,y,z = ez.shape
        for i in prange(1, x):
            for j in prange(1, y):
                for k in prange(1, z):
                    iez[i, j, k] = iez[i, j, k] + (hy[i, j, k] - hy[i - 1, j, k] - hx[i, j, k] + hx[i, j - 1, k])
                    curl_e = pdx3[i] * pdy3[j] * ez[i, j, k] \
                        + pdx2[i] * pdy2[j] * (0.5 * (hy[i, j, k] - hy[i - 1, j, k] - hx[i, j, k] + hx[i, j - 1, k]) + pdz1[k] * iez[i, j, k])
                    idx = media[i,j,k]
                    ez[i, j, k] = cez[idx] * (curl_e - iz[i, j, k])
                    iz[i, j, k] = iz[i, j, k] + cezl[idx] * ez[i, j, k]
        return ez, iz, iez


    @staticmethod
    @njit(parallel=True)
    def calculate_hx_field(hx,ey,ez):
        x,y,z = hx.shape
        for i in prange(0, x):
            for j in prange(0, y - 1):
                for k in prange(0, z - 1):
                    hx[i, j, k] = hx[i, j, k] + 0.5 * (ey[i, j, k + 1] - ey[i, j, k] - ez[i, j + 1, k] + ez[i, j, k])
        return hx


    @staticmethod
    @njit(parallel=True)
    def calculate_hy_field(hy,ex,ez):
        x,y,z = hy.shape
        for i in prange(0, x-1):
            for j in prange(0, y):
                for k in prange(0, z-1):
                    hy[i, j, k] = hy[i, j, k] + 0.5 * (ez[i + 1, j, k] - ez[i, j, k] - ex[i, j, k + 1] + ex[i, j, k])
        return hy


    @staticmethod
    @njit(parallel=True)
    def calculate_hz_field(hz, ex, ey):
        x,y,z = hz.shape
        for i in prange(0, x-1):
            for j in prange(0, y-1):
                for k in prange(0, z):
                    hz[i, j, k] = hz[i, j, k] + 0.5 * (ex[i, j + 1, k] - ex[i, j, k] - ey[i + 1, j, k] + ey[i, j, k])
        return hz


    @staticmethod
    @njit(parallel=True)
    def calculate_hx_field_pml(hx,ey,ez,ihx,phx1,phy2,phy3,phz2,phz3):
        x,y,z = hx.shape
        for i in prange(0, x):
            for j in prange(0, y - 1):
                for k in prange(0, z - 1):
                    ihx[i, j, k] = ihx[i, j, k] + (ey[i, j, k + 1] - ey[i, j, k] - ez[i, j + 1, k] + ez[i, j, k])
                    hx[i, j, k] = phy3[j] * phz3[k] * hx[i, j, k] \
                        + phy2[j] * phz2[k] * 0.5 * ((ey[i, j, k + 1] - ey[i, j, k] - ez[i, j + 1, k] + ez[i, j, k]) + phx1[i] * ihx[i, j, k])
        return hx, ihx


    @staticmethod
    @njit(parallel=True)
    def calculate_hy_field_pml(hy,ex,ez,ihy,phx2,phx3,phy1,phz2,phz3):
        x,y,z = hy.shape
        for i in prange(0, x-1):
            for j in prange(0, y):
                for k in prange(0, z-1):
                    ihy[i, j, k] = ihy[i, j, k] + (ez[i + 1, j, k] - ez[i, j, k] - ex[i, j, k + 1] + ex[i, j, k])
                    hy[i, j, k] = phx3[i] * phz3[k] * hy[i, j, k] \
                        + phx2[i] * phz2[k] * 0.5 * ((ez[i + 1, j, k] - ez[i, j, k] - ex[i, j, k + 1] + ex[i, j, k]) + phy1[j] * ihy[i, j, k])
        return hy, ihy


    @staticmethod
    @njit(parallel=True)
    def calculate_hz_field_pml(hz, ex, ey,ihz,phx2,phx3,phy2,phy3,phz1):
        x,y,z = hz.shape
        for i in prange(0, x-1):
            for j in prange(0, y-1):
                for k in prange(0, z):
                    ihz[i, j, k] = ihz[i, j, k] + (ex[i, j + 1, k] - ex[i, j, k] - ey[i + 1, j, k] + ey[i, j, k])
                    hz[i, j, k] = phx3[i] * phy3[j] * hz[i, j, k] \
                        + phx2[i] * phy2[j] * 0.5 * ((ex[i, j + 1, k] - ex[i, j, k] - ey[i + 1, j, k] + ey[i, j, k]) + phz1[k] * ihz[i, j, k])
        return hz, ihz


    def update_e_fields(self):
        if self.boundaries == 'None':
            self.ex,self.ix = self.calculate_ex_field(self.ex,self.hy,self.hz,self.ix,self.cex,self.cexl,self.media)
            self.ey,self.iy = self.calculate_ey_field(self.ey,self.hx,self.hz,self.iy,self.cey,self.ceyl,self.media)
            self.ez,self.iz = self.calculate_ez_field(self.ez,self.hx,self.hy,self.iz,self.cez,self.cezl,self.media)

        elif self.boundaries == 'PML':
            self.ex,self.ix,self.boundary.iex = self.calculate_ex_field_pml(self.ex,self.hy,self.hz,self.ix,self.cex,self.cexl,self.media,self.boundary.iex,self.boundary.pdx1,self.boundary.pdy2,self.boundary.pdy3,self.boundary.pdz2,self.boundary.pdz3)
            self.ey,self.iy,self.boundary.iey = self.calculate_ey_field_pml(self.ey,self.hx,self.hz,self.iy,self.cey,self.ceyl,self.media,self.boundary.iey,self.boundary.pdx2,self.boundary.pdx3,self.boundary.pdy1,self.boundary.pdz2,self.boundary.pdz3)
            self.ez,self.iz,self.boundary.iez = self.calculate_ez_field_pml(self.ez,self.hx,self.hy,self.iz,self.cez,self.cezl,self.media,self.boundary.iez,self.boundary.pdx2,self.boundary.pdx3,self.boundary.pdy2,self.boundary.pdy3,self.boundary.pdz1)

        else:
            print('err')


    def update_h_fields(self):

        if self.boundaries == 'None':
            self.hx = self.calculate_hx_field(self.hx,self.ey,self.ez)
            self.hy = self.calculate_hy_field(self.hy,self.ex,self.ez)
            self.hz = self.calculate_hz_field(self.hz,self.ex,self.ey)

        elif self.boundaries == 'PML':
            self.hx,self.boundary.ihx = self.calculate_hx_field_pml(self.hx,self.ey,self.ez,self.boundary.ihx,self.boundary.phx1,self.boundary.phy2,self.boundary.phy3,self.boundary.phz2,self.boundary.phz3)
            self.hy,self.boundary.ihy = self.calculate_hy_field_pml(self.hy,self.ex,self.ez,self.boundary.ihy,self.boundary.phx2,self.boundary.phx3,self.boundary.phy1,self.boundary.phz2,self.boundary.phz3)
            self.hz,self.boundary.ihz = self.calculate_hz_field_pml(self.hz,self.ex,self.ey,self.boundary.ihz,self.boundary.phx2,self.boundary.phx3,self.boundary.phy2,self.boundary.phy3,self.boundary.phz1)
        else:
            print('err')


    def run(self):
        self.ez[self.nxc,self.nyc,self.nzc] = self.source.update_source()
        self.update_e_fields()
        self.ez[self.nxc,self.nyc,self.nzc] = self.source.update_source()
        self.update_h_fields()


    def run_animation(self,nsteps): # -> matplotlib.animation.ArtistAnimation
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
            amp = self.calculate_amp(self.ex,self.ey,self.ez)
            im1 = ax.imshow(amp[:,:,self.nxc].T, cmap='jet', origin='lower',vmax=1e-2)
            im2 = ax.imshow(masked_where(self.media[:,:,self.nxc].T == 0, self.media[:,:,self.nxc].T),interpolation='nearest',cmap=custom_map,alpha=0.5)
            ims.append([im1,im2])
            self.run()

        cbar = fig.colorbar(im1)
        cbar.ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        plt.close()
        plt.rcParams['animation.embed_limit'] = 50
        ani = animation.ArtistAnimation(fig, ims, interval=50, blit=False)

        return ani

