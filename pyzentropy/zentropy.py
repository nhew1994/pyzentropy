from scipy import constants
import numpy as np
from scipy.interpolate import UnivariateSpline

BOLTZMANN_CONSTANT = constants.physical_constants[
    "Boltzmann constant in eV/K"
][0]

class InternalEnergy:
    def __init__(
        self,
        internal_energy: np.array,
        volume: np.array
    ):
        self.energy = internal_energy
        self.volume = volume

class HelmholtzEnergy:
    def __init__(
        self,
        helmholtz_energy: np.array,
        volume: np.array,
        temperature: np.array,
    ):
        self.energy = helmholtz_energy
        self.volume = volume
        self.temperature = temperature

class Entropy:
    def __init__(
        self,
        entropy,
        volume,
        temperature
    ):
        self.entropy = entropy
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
        probability = None
    ):
        self.config_name = config_name
        self.structure = structure
        self.internal_energy = internal_energy
        self.entropy = entropy
        self.helmholtz_energy = helmholtz_energy
        
    def helmolholtz_energy(
        self,
        internal_energy: InternalEnergy,
        entropy: Entropy
    ):
        if internal_energy.volume != entropy.volume:
            raise ValueError(
                "Internal energy and entropy volumes must be the same"
            )
        u = internal_energy.energy
        v = internal_energy.volume
        t = entropy.temperature
        s = entropy.entropy
        
        f = np.zeros((len(t), len(v)))
        for i in enumerate(t):
            f[i] = u - t*s[i]
        self.helmholtz_energy = HelmholtzEnergy(
            helmholtz_energy = f,
            volume = v,
            temperature = t
        )

    def internal_energy(
        self,
        helmholtz_energy: HelmholtzEnergy,
        entropy: Entropy
    ):
        if helmholtz_energy.volume != entropy.volume:
            raise ValueError(
                "Helmholtz energy and entropy volumes must be the same"
            )
        if helmholtz_energy.temperature != entropy.temperature:
            raise ValueError(
                "Helmholtz energy and entropy temperatures must be the same"
            )
        
        f = helmholtz_energy.energy
        v = helmholtz_energy.volume
        t = entropy.temperature
        s = entropy.entropy
        u = f[0] + t[0]*s[0]
        self.internal_energy = InternalEnergy(
            internal_energy = u,
            volume = v
        )
    
    def entropy(
        self,
        helmholtz_energy: HelmholtzEnergy,
        internal_energy: InternalEnergy
    ):
        if helmholtz_energy.volume != internal_energy.volume:
            raise ValueError(
                "Helmholtz energy and internal energy volumes must be the same"
            )
        if helmholtz_energy.temperature != internal_energy.temperature:
            raise ValueError(
                "Helmholtz energy and internal energy temperatures must be the same"
            )
        
        f = helmholtz_energy.energy
        v = helmholtz_energy.volume
        t = helmholtz_energy.temperature
        u = internal_energy.energy

        for i in enumerate(t):
            s = (f[i] - u[i])/t[i]
        self.entropy = Entropy(
            entropy = s,
            volume = v,
            temperature = t
        )

class System:
    def __init__(
        self,
        configurations,
    ):
        self.configurations = configurations
    def partition_function(self):
        config_helmholtz_energies = np.array(
            [config.helmholtz_energy.energy for config in self.configurations]
        )
        self.helmholtz_energy = None
        self.gibbs_entropy = None
        self.inter_entropy = self.gibbs_entropy
        self.intra_entropy = None
        self.entropy = None
        self.bulk_modulus = None
        self.inter_bulk_modulus = None
        self.intra_bulk_modulus = None
        self.alt_system_bulk_modulus = None
        self.intra_heat_capacity = None
        self.inter_heat_capacity = None
        self.alt_heat_capacity = None
        self.heat_capacity = None

