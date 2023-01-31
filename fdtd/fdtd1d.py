from .grid import Grid
from .boundaries import PML
from .sources import PointSource

from numba import njit,prange
from numpy import float32

class FDTD1d(Grid):
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
    def calculate_dx_field(dx,hy):
        x, = dx.shape
        for i in prange(1,x):
            dx[i] = dx[i] + 0.5 * (hy[i - 1] - hy[i])

        return dx


    @staticmethod
    @njit(parallel=True)
    def calculate_ex_field(ex,ix,dx,cex,cexl,media):
        x, = ex.shape
        for i in prange(1, x):
            idx =media[i]
            ex[i] = cex[idx] * (dx[i] - ix[i])
            ix[i] = ix[i] + cexl[idx] * ex[i]

        return ex,ix


    @staticmethod
    @njit(parallel=True)
    def calculate_hy_field(hy,ex):
        x, = hy.shape
        for i in prange(x - 1):
            hy[i] = hy[i] + 0.5 * (ex[i] - ex[i + 1])

        return hy


    def update_d_fields(self):
        self.dx = self.calculate_dx_field(self.dx,self.hy)


    def update_e_fields(self):
        self.ex,self.ix = self.calculate_ex_field(self.ex,self.ix,self.dx,self.cex,self.cexl,self.media)


    def update_h_fields(self):
        self.hy = self.calculate_hy_field(self.hy,self.ex)


    def run(self):
        self.update_d_fields()
        self.dx[self.nxc] = self.source.update_source()
        self.update_e_fields()
        self.dx[self.nxc] = self.source.update_source()
        self.update_h_fields()

    def run_animation(self,nsteps):
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        ims = []
        fig,ax = plt.subplots()
        for _ in range(nsteps):
            im = ax.plot(self.ex,c='blue')
            ims.append(im)
            self.run()
        plt.close()
        plt.rcParams['animation.embed_limit'] = 50
        ani = animation.ArtistAnimation(fig, ims, interval=50, blit=False)

        return ani
