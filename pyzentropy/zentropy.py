from scipy import constants
import numpy as np
from functools import cached_property
from scipy.interpolate import UnivariateSpline

BOLTZMANN_CONSTANT = constants.physical_constants[
    "Boltzmann constant in eV/K"
][0]

class InternalEnergyOfV:
    def __init__(
        self,
        internal_energy: np.array,
        volume: np.array
    ):
        self.energy = internal_energy
        self.volume = volume

class EntropyOfVT:
    def __init__(
        self,
        entropy,
        volume,
        temperature
    ):
        self.entropy = entropy
        self.volume = volume
        self.temperature = temperature

class HelmholtzEnergyOfVT:
    def __init__(
        self,
        helmholtz_energy: np.array,
        volume: np.array,
        temperature: np.array,
    ):
        self.energy = helmholtz_energy
        self.volume = volume
        self.temperature = temperature

class Configuration:
    def __init__(
        self,
        config_name,
        structure,
        internal_energy = None,
        entropy = None,
        helmholtz_energy = None,
        partition_function = None,
    ):
        self.config_name = config_name
        self.structure = structure
        self.internal_energy = internal_energy
        self.entropy = entropy
        self.helmholtz_energy = helmholtz_energy
        self.partition_function = partition_function

class System:
    def __init__(
        self,
        name,
        configurations,
        internal_energy = None,
        entropy = None,
        helmholtz_energy = None,
        partition_function = None,
    ):
        self.name = name
        self.configurations = configurations
        self.internal_energy = internal_energy
        self.entropy = entropy
        self.helmholtz_energy = helmholtz_energy
        self.partition_function = partition_function
        
