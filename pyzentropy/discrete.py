from scipy import constants
import numpy as np
from functools import cached_property
from scipy.interpolate import UnivariateSpline

BOLTZMANN_CONSTANT = constants.physical_constants[
    "Boltzmann constant in eV/K"
][0]


class DiscreteV:
    def __init__(
        self,
        value: np.array,
        volume: np.array
    ):
        self.value = value
        self.volume = volume
        
class DiscreteT:
    def __init__(
        self,
        value: np.array,
        temperature: np.array
    ):
        self.value = value
        self.temperature = temperature

class DiscreteVT:
    def __init__(
        self,
        value: np.array,
        volume: np.array,
        temperature: np.array
    ):
        self.value = value
        self.volume = volume
        self.temperature = temperature