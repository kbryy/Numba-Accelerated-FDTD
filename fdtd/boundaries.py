import numpy as np

class PML:
    def __init__(self,shape,DTYPE):
        self.DTYPE = DTYPE
        d = len(shape)

        if d == 2:
            self.nx,self.ny = shape
            self.ihx   = np.zeros((self.nx,self.ny),dtype=self.DTYPE)
            self.ihy   = np.zeros((self.nx,self.ny),dtype=self.DTYPE)
            self.calculate_pml_coefficients_2d()

        elif d == 3:
            self.nx,self.ny,self.nz = shape
            self.iex = np.zeros((self.nx,self.ny,self.nz),dtype=self.DTYPE)
            self.iey = np.zeros((self.nx,self.ny,self.nz),dtype=self.DTYPE)
            self.iez = np.zeros((self.nx,self.ny,self.nz),dtype=self.DTYPE)
            self.ihx = np.zeros((self.nx,self.ny,self.nz),dtype=self.DTYPE)
            self.ihy = np.zeros((self.nx,self.ny,self.nz),dtype=self.DTYPE)
            self.ihz = np.zeros((self.nx,self.ny,self.nz),dtype=self.DTYPE)
            self.calculate_pml_coefficients_3d()


    def calculate_pml_coefficients_2d(self):
        pdx2 = np.ones(self.nx,dtype=self.DTYPE)
        pdx3 = np.ones(self.nx,dtype=self.DTYPE)
        pdy2 = np.ones(self.ny,dtype=self.DTYPE)
        pdy3 = np.ones(self.ny,dtype=self.DTYPE)

        phx1 = np.zeros(self.nx,dtype=self.DTYPE)
        phx2 = np.ones(self.nx,dtype=self.DTYPE)
        phx3 = np.ones(self.nx,dtype=self.DTYPE)
        phy1 = np.zeros(self.ny,dtype=self.DTYPE)
        phy2 = np.ones(self.ny,dtype=self.DTYPE)
        phy3 = np.ones(self.ny,dtype=self.DTYPE)

        npml = 12
        for n in range(npml):
            xnum = npml - n
            xd = npml
            xxn = xnum / xd
            xn = 0.33 * xxn ** 3
            pdx2[n] = 1 / (1 + xn)
            pdx2[self.nx - 1 - n] = 1 / (1 + xn)
            pdx3[n] = (1 - xn) / (1 + xn)
            pdx3[self.nx - 1 - n] = (1 - xn) / (1 + xn)
            pdy2[n] = 1 / (1 + xn)
            pdy2[self.ny - 1 - n] = 1 / (1 + xn)
            pdy3[n] = (1 - xn) / (1 + xn)
            pdy3[self.ny - 1 - n] = (1 - xn) / (1 + xn)

            xxn = (xnum - 0.5) / xd
            xn = 0.33 * xxn ** 3
            phx1[n] = xn
            phx1[self.nx - 2 - n] = xn
            phx2[n] = 1 / (1 + xn)
            phx2[self.nx - 2 - n] = 1 / (1 + xn)
            phx3[n] = (1 - xn) / (1 + xn)
            phx3[self.nx - 2 - n] = (1 - xn) / (1 + xn)
            phy1[n] = xn
            phy1[self.ny - 2 - n] = xn
            phy2[n] = 1 / (1 + xn)
            phy2[self.ny - 2 - n] = 1 / (1 + xn)
            phy3[n] = (1 - xn) / (1 + xn)
            phy3[self.ny - 2 - n] = (1 - xn) / (1 + xn)

        self.pdx2 = pdx2
        self.pdx3 = pdx3
        self.pdy2 = pdy2
        self.pdy3 = pdy3

        self.phx1 = phx1
        self.phx2 = phx2
        self.phx3 = phx3
        self.phy1 = phy1
        self.phy2 = phy2
        self.phy3 = phy3


    def calculate_pml_coefficients_3d(self):
        pdx1 = np.zeros(self.nx)
        pdx2 = np.ones(self.nx)
        pdx3 = np.ones(self.nx)
        pdy1 = np.zeros(self.ny)
        pdy2 = np.ones(self.ny)
        pdy3 = np.ones(self.ny)
        pdz1 = np.zeros(self.nz)
        pdz2 = np.ones(self.nz)
        pdz3 = np.ones(self.nz)

        phx1 = np.zeros(self.nx)
        phx2 = np.ones(self.nx)
        phx3 = np.ones(self.nx)
        phy1 = np.zeros(self.ny)
        phy2 = np.ones(self.ny)
        phy3 = np.ones(self.ny)
        phz1 = np.zeros(self.nz)
        phz2 = np.ones(self.nz)
        phz3 = np.ones(self.nz)

        npml = 12
        for n in range(npml):
            xxn = (npml - n) / npml
            xn = 0.33 * (xxn ** 3)
            pdx2[n] = 1 / (1 + xn)
            pdy2[n] = 1 / (1 + xn)
            pdz2[n] = 1 / (1 + xn)
            pdx2[self.nx-1-n] = 1 / (1 + xn)
            pdy2[self.ny-1-n] = 1 / (1 + xn)
            pdz2[self.nz-1-n] = 1 / (1 + xn)
            pdx3[n] = (1 - xn) / (1 + xn)
            pdy3[n] = (1 - xn) / (1 + xn)
            pdz3[n] = (1 - xn) / (1 + xn)
            pdx3[self.nx-1-n] = (1 - xn) / (1 + xn)
            pdy3[self.ny-1-n] = (1 - xn) / (1 + xn)
            pdz3[self.nz-1-n] = (1 - xn) / (1 + xn)
            phx1[n] = xn
            phy1[n] = xn
            phz1[n] = xn
            phx1[self.nx-n-1] = xn
            phy1[self.ny-n-1] = xn
            phz1[self.nz-n-1] = xn

            xxn = (npml - n - 0.5) / npml
            xn = 0.33 * (xxn ** 3)
            pdx1[n] = xn
            pdy1[n] = xn
            pdz1[n] = xn
            pdx1[self.nx-1-n] = xn
            pdy1[self.ny-1-n] = xn
            pdz1[self.nz-1-n] = xn
            phx2[n] = 1 / (1 + xn)
            phy2[n] = 1 / (1 + xn)
            phz2[n] = 1 / (1 + xn)
            phx2[self.nx-1-n] = 1 / (1 + xn)
            phy2[self.ny-1-n] = 1 / (1 + xn)
            phz2[self.nz-1-n] = 1 / (1 + xn)
            phx3[n] = (1 - xn) / (1 + xn)
            phy3[n] = (1 - xn) / (1 + xn)
            phz3[n] = (1 - xn) / (1 + xn)
            phx3[self.nx-1-n] = (1 - xn) / (1 + xn)
            phy3[self.ny-1-n] = (1 - xn) / (1 + xn)
            phz3[self.nz-1-n] = (1 - xn) / (1 + xn)

        self.pdx1 = pdx1
        self.pdx2 = pdx2
        self.pdx3 = pdx3
        self.pdy1 = pdy1
        self.pdy2 = pdy2
        self.pdy3 = pdy3
        self.pdz1 = pdz1
        self.pdz2 = pdz2
        self.pdz3 = pdz3

        self.phx1 = phx1
        self.phx2 = phx2
        self.phx3 = phx3
        self.phy1 = phy1
        self.phy2 = phy2
        self.phy3 = phy3
        self.phz1 = phz1
        self.phz2 = phz2
        self.phz3 = phz3
