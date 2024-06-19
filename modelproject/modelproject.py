from types import SimpleNamespace
import time
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

class modelclass():

    def __init__(self):

        par = self.par = SimpleNamespace()

        par.rho = 0.05
        par.A = 1
        par.alpha = 1/3
        par.n = 0.04
        par.tau = 0

    
