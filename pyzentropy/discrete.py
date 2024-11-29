from scipy import constants
import numpy as np
from functools import cached_property
from scipy.interpolate import UnivariateSpline

BOLTZMANN_CONSTANT = constants.physical_constants[
    "Boltzmann constant in eV/K"
][0]    

class Discrete:
    def __init__(self, value: np.array, **kwargs):
        self.value = value
        for key, val in kwargs.items():
            setattr(self, key, val)

class DiscreteV(Discrete):
    def __init__(self, value: np.array, volume: np.array):
        super().__init__(value, volume=volume)

class DiscreteT(Discrete):
    def __init__(self, value: np.array, temperature: np.array):
        super().__init__(value, temperature=temperature)

class DiscreteVT(Discrete):
    def __init__(self, value: np.array, volume: np.array, temperature: np.array):
        super().__init__(value, volume=volume, temperature=temperature)

class DiscreteN(Discrete):
    def __init__(self, value: np.array, number: np.array):
        super().__init__(value, number=number)

class DiscreteNV(Discrete):
    def __init__(self, value: np.array, number: np.array, volume: np.array):
        super().__init__(value, number=number, volume=volume)

class DiscreteNT(Discrete):
    def __init__(self, value: np.array, number: np.array, temperature: np.array):
        super().__init__(value, number=number, temperature=temperature)

class DiscreteNVT(Discrete):
    def __init__(self, value: np.array, number: np.array, volume: np.array, temperature: np.array):
        super().__init__(value, number=number, volume=volume, temperature=temperature)

class Configuration:
    def __init__(
        self,
        config_name,
        structure,
        internal_energy = None,
        entropy = None,
        helmholtz_energy = None,
        partition_function = None,
        heat_capacity = None,
        bulk_modulus = None,
        equation_of_state = None,
        equation_of_state_parameters = None
    ):
        self.config_name = config_name
        self.structure = structure
        self.internal_energy = internal_energy
        self.entropy = entropy
        self.helmholtz_energy = helmholtz_energy
        self.partition_function = partition_function
        self.heat_capacity = heat_capacity
        self.bulk_modulus = bulk_modulus
        self.equation_of_state = equation_of_state
        self.equation_of_state_parameters = equation_of_state_parameters

class System: