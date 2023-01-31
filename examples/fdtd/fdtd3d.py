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
    def calculate_dx_field(dx,hy,hz):
        x,y,z = dx.shape
        for i in prange(1, x):
            for j in prange(1, y):
                for k in prange(1, z):
                    dx[i, j, k] = dx[i, j, k] + 0.5 * (hz[i, j, k] - hz[i, j - 1, k] - hy[i, j, k] + hy[i, j, k - 1])
        return dx


    @staticmethod
    @njit(parallel=True)
    def calculate_dy_field(dy,hx,hz):
        x,y,z = dy.shape
        for i in prange(1, x):
            for j in prange(1, y):
                for k in prange(1, z):
                    dy[i, j, k] = dy[i, j, k] + 0.5 * (hx[i, j, k] - hx[i, j, k - 1] - hz[i, j, k] + hz[i - 1, j, k])
        return dy


    @staticmethod
    @njit(parallel=True)
    def calculate_dz_field(dz,hx,hy):
        x,y,z = dz.shape
        for i in prange(1, x):
            for j in prange(1, y):
                for k in prange(1, z):
                    dz[i, j, k] = dz[i, j, k] + 0.5 * (hy[i, j, k] - hy[i - 1, j, k] - hx[i, j, k] + hx[i, j - 1, k])
        return dz


    @staticmethod
    @njit(parallel=True)
    def calculate_dx_field_pml(dx,hy,hz,idx,pdx1,pdy2,pdy3,pdz2,pdz3):
        x,y,z = dx.shape
        for i in prange(1, x):
            for j in prange(1, y):
                for k in prange(1, z):
                    idx[i, j, k] = idx[i, j, k] + (hz[i, j, k] - hz[i, j - 1, k] - hy[i, j, k] + hy[i, j, k - 1])
                    dx[i, j, k] = pdy3[j] * pdz3[k] * dx[i, j, k] \
                        + pdy2[j] * pdz2[k] * (0.5 * (hz[i, j, k] - hz[i, j - 1, k] - hy[i, j, k] + hy[i, j, k - 1]) + pdx1[i] * idx[i, j, k])
        return dx, idx


    @staticmethod
    @njit(parallel=True)
    def calculate_dy_field_pml(dy,hx,hz,idy,pdx2,pdx3,pdy1,pdz2,pdz3):
        x,y,z = dy.shape
        for i in prange(1, x):
            for j in prange(1, y):
                for k in prange(1, z):
                    idy[i, j, k] = idy[i, j, k] + (hx[i, j, k] - hx[i, j, k - 1] - hz[i, j, k] + hz[i - 1, j, k])
                    dy[i, j, k] = pdx3[i] * pdz3[k] * dy[i, j, k] \
                        + pdx2[i] * pdz2[k] * (0.5 * (hx[i, j, k] - hx[i, j, k - 1] - hz[i, j, k] + hz[i - 1, j, k]) + pdy1[j] * idy[i, j, k])
        return dy, idy


    @staticmethod
    @njit(parallel=True)
    def calculate_dz_field_pml(dz,hx,hy,idz,pdx2,pdx3,pdy2,pdy3,pdz1):
        x,y,z = dz.shape
        for i in prange(1, x):
            for j in prange(1, y):
                for k in prange(1, z):
                    idz[i, j, k] = idz[i, j, k] + (hy[i, j, k] - hy[i - 1, j, k] - hx[i, j, k] + hx[i, j - 1, k])
                    dz[i, j, k] = pdx3[i] * pdy3[j] * dz[i, j, k] \
                        + pdx2[i] * pdy2[j] * (0.5 * (hy[i, j, k] - hy[i - 1, j, k] - hx[i, j, k] + hx[i, j - 1, k]) + pdz1[k] * idz[i, j, k])
        return dz, idz


    @staticmethod
    @njit(parallel=True)
    def calculate_ex_field(ex,ix,dx,cex,cexl,media):
        x,y,z = ex.shape
        for i in prange(0, x):
            for j in prange(0, y):
                for k in prange(0, z):
                    idx = media[i,j,k]
                    ex[i, j, k] = cex[idx] * (dx[i, j, k] - ix[i, j, k])
                    ix[i, j, k] = ix[i, j, k] + cexl[idx] * ex[i, j, k]
        return ex,ix

    @staticmethod
    @njit(parallel=True)
    def calculate_ey_field(ey,iy,dy,cey,ceyl,media):
        x,y,z = ey.shape
        for i in prange(0, x):
            for j in prange(0, y):
                for k in prange(0, z):
                    idx = media[i,j,k]
                    ey[i, j, k] = cey[idx] * (dy[i, j, k] - iy[i, j, k])
                    iy[i, j, k] = iy[i, j, k] + ceyl[idx] * ey[i, j, k]
        return ey,iy

    @staticmethod
    @njit(parallel=True)
    def calculate_ez_field(ez,iz,dz,cez,cezl,media):
        x,y,z = ez.shape
        for i in prange(0, x):
            for j in prange(0, y):
                for k in prange(0, z):
                    idx = media[i,j,k]
                    ez[i, j, k] = cez[idx] * (dz[i, j, k] - iz[i, j, k])
                    iz[i, j, k] = iz[i, j, k] + cezl[idx] * ez[i, j, k]

        return ez,iz


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


    def update_d_fields(self):

        if self.boundaries == 'None':
            self.dx = self.calculate_dx_field(self.dx,self.hy,self.hz)
            self.dy = self.calculate_dy_field(self.dy,self.hx,self.hz)
            self.dz = self.calculate_dz_field(self.dz,self.hx,self.hy)

        elif self.boundaries == 'PML':
            self.dx,self.boundary.idx = self.calculate_dx_field_pml(self.dx,self.hy,self.hz,self.boundary.idx,self.boundary.pdx1,self.boundary.pdy2,self.boundary.pdy3,self.boundary.pdz2,self.boundary.pdz3)
            self.dy,self.boundary.idy = self.calculate_dy_field_pml(self.dy,self.hx,self.hz,self.boundary.idy,self.boundary.pdx2,self.boundary.pdx3,self.boundary.pdy1,self.boundary.pdz2,self.boundary.pdz3)
            self.dz,self.boundary.idz = self.calculate_dz_field_pml(self.dz,self.hx,self.hy,self.boundary.idz,self.boundary.pdx2,self.boundary.pdx3,self.boundary.pdy2,self.boundary.pdy3,self.boundary.pdz1)
        else:
            print('err')

    def update_e_fields(self):

        self.ex,self.ix = self.calculate_ex_field(self.ex,self.ix,self.dx,self.cex,self.cexl,self.media)
        self.ey,self.iy = self.calculate_ey_field(self.ey,self.iy,self.dy,self.cey,self.ceyl,self.media)
        self.ez,self.iz = self.calculate_ez_field(self.ez,self.iz,self.dz,self.cez,self.cezl,self.media)


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
        self.update_d_fields()
        self.dz[self.nxc,self.nyc,self.nzc] = self.source.update_source()
        self.update_e_fields()
        self.dz[self.nxc,self.nyc,self.nzc] = self.source.update_source()
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

