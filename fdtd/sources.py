from math import pi
from numpy import sin
from numba import njit

class PointSource:
    def __init__(self,f,dt):
        self.dt = dt
        self.f  = f
        self.t  = 0.0


    def update_source(self):
        result  = self.calculate_sin_signal_at_time(self.f,self.t)
        self.t += self.dt / 2.0

        return result


    @staticmethod
    @njit
    def calculate_sin_signal_at_time(f,t):
        return sin(2 * pi * f * t)
