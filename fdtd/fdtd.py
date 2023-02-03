from numpy import float32

from .fdtd1d import FDTD1d
from .fdtd2d import FDTD2d
from .fdtd3d import FDTD3d

def fdtd(media,constants,ddx,f,DTYPE=float32):
    if media.ndim == 1:
        return FDTD1d(media,constants,ddx,f,DTYPE)

    elif media.ndim == 2:
        return FDTD2d(media,constants,ddx,f,DTYPE)

    elif media.ndim == 3:
        return FDTD3d(media,constants,ddx,f,DTYPE)
