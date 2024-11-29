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
        volume = None,
        temperature = None,
        internal_energy = None,
        entropy = None,
        helmholtz_energy = None
    ):
        self.config_name = config_name
        self.structure = structure
        self.volume = volume
        self.temperature = temperature
        self.internal_energy = internal_energy
        self.entropy = entropy
        self.helmholtz_energy = helmholtz_energy
    
    @property
    def volume(self):
        return self._volume
    
    @volume.setter
    def volume(self, value):
        self._volume = value

    @volume.getter
    def volume(self):
        return self._volume
    
    @volume.deleter
    def volume(self):
        del self._volume
    
    @property
    def temperature(self):
        return self._temperature
    
    @temperature.setter
    def temperature(self, value):
        self._temperature = value

    @temperature.getter
    def temperature(self):
        return self._temperature
    
    @temperature.deleter
    def temperature(self):
        del self._temperature

    @property
    def internal_energy(self):
        return self._internal_energy
    
    @internal_energy.setter
    def internal_energy(self, internal_energy):
        if self._entropy is not None and \
            self._volume != internal_energy.volume:
            raise ValueError(
                "Configuration entropy is defined and the configuration "
                "volumes are inconsistent with those of the given "
                "InternalEnergy object"
            )
        if self._helmholtz_energy is not None and \
            self._volume != internal_energy.volume:
            raise ValueError(
                "Configuration Helmholtz energy is defined and the "
                "Configuration volumes are inconsistent with those of the "
                "given InternalEnergy object"
            )
        
        self._internal_energy = internal_energy.energy

        if self.volume != internal_energy.volume:
            self._volume = internal_energy.volume
    
    @internal_energy.getter
    def internal_energy(self):
        return self._internal_energy
    
    @internal_energy.deleter
    def internal_energy(self):
        del self._internal_energy

    @property
    def entropy(self):
        return self._entropy
    
    @entropy.setter
    def entropy(self, entropy):
        if self._internal_energy is not None:
            if self._volume != entropy.volume:
                raise ValueError(
                    "Configuration internal energy is defined and the "
                    "Configuration volumes are inconsistent with those of the "
                    "given Entropy object"
                )
            if self._temperature != entropy.temperature:
                raise ValueError(
                    "Configuration internal energy is defined and the "
                    "Configuration temperatures are inconsistent with those "
                    "of the given Entropy object"
                )
        
        if self._helmholtz_energy is not None:
            if self._volume != entropy.volume:
                raise ValueError(
                    "Configuration Helmholtz energy is defined and the "
                    "Configuration volumes are inconsistent with those of the "
                    "given Entropy object"
                )
            if self._temperature != entropy.temperature:
                raise ValueError(
                    "Configuration Helmholtz energy is defined and the "
                    "Configuration temperatures are inconsistent with those "
                    "of the given Entropy object"
                )

        self._entropy = entropy.entropy
        if self.volume != entropy.volume:
            self._volume = entropy.volume
        if self.temperature != entropy.temperature:
            self._temperature = entropy.temperature

    @entropy.getter
    def entropy(self):
        return self._entropy
    
    @entropy.deleter
    def entropy(self):
        del self._entropy

    @property
    def helmholtz_energy(self):
        return self._helmholtz_energy
    
    @helmholtz_energy.setter
    def helmholtz_energy(self, helmholtz_energy):
        if self._internal_energy is not None:
            if self._volume != helmholtz_energy.volume:
                raise ValueError(
                    "Configuration internal energy is defined and "
                    "Configuration volume is inconsistent with the volume of "
                    "the given HelmholtzEnergy object"
                )
            if self._temperature != helmholtz_energy.temperature:
                raise ValueError(
                    "Configuration internal energy is defined and "
                    "Configuration temperature is inconsistent with the "
                    "temperature of the given HelmholtzEnergy object"
                )
        
        if self._entropy is not None:
            if self._volume != helmholtz_energy.volume:
                raise ValueError(
                    "Configuration entropy is defined and Configuration "
                    "volume is inconsistent with the volume of the given "
                    "HelmholtzEnergy object"
                )
            if self._temperature != helmholtz_energy.temperature:
                raise ValueError(
                    "Configuration entropy is defined and Configuration "
                    "temperature is inconsistent with the temperature of the "
                    "given HelmholtzEnergy object"
                )

        self._helmholtz_energy = helmholtz_energy.energy
        if self.volume != helmholtz_energy.volume:
            self._volume = helmholtz_energy.volume
        if self.temperature != helmholtz_energy.temperature:
            self._temperature = helmholtz_energy.temperature
    
    @helmholtz_energy.getter
    def helmholtz_energy(self):
        return self._helmholtz_energy


    @property
    def helmholtz_energy(self):
        if self.volume != entropy.volume:
            raise ValueError(
                "Internal energy and entropy volumes must be the same"
            )
        u = self.internal_energy
        v = self.volume
        t = self.temperature
        s = self.entropy
        
        f = np.zeros((len(t), len(v)))
        for i in enumerate(t):
            f[i] = u - t*s[i]
        self.helmholtz_energy = HelmholtzEnergy(
            helmholtz_energy = f,
            volume = v,
            temperature = t
        )
    @helmholtz_energy.setter
    def helmholtz_energy(self, value):
        self.helmholtz_energy = value

    @helmholtz_energy.getter
    def helmholtz_energy(self):
        return self.helmholtz_energy
    
    @helmholtz_energy.deleter
    def helmholtz_energy(self):
        del self.helmholtz_energy
    
    

    def partition_function(
        self,
        helmholtz_energy: HelmholtzEnergy,
    ):
        f = helmholtz_energy.energy
        v = helmholtz_energy.volume
        t = helmholtz_energy.temperature
        beta = 1/(BOLTZMANN_CONSTANT*t)
        zk = np.zeros((len(t), len(v)))
        for i in enumerate(t):
            zk[i] = np.exp(
                -f[i]*beta
            )
        return zk

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
        partition_function = None
    ):
        self.configurations = configurations
    def partition_function(self):
        # check if all configurations have the same volume and temperature
        config_volumes = [
            config.helmholtz_energy.volume for config in self.configurations
        ]
        config_temperatures = [
            config.helmholtz_energy.temperature for config in self.configurations
        ]
        if not all([v == config_volumes[0] for v in config_volumes]):
            raise ValueError("Configurations must have the same volume")
        if not all([t == config_temperatures[0] for t in config_temperatures]):
            raise ValueError("Configurations must have the same temperature")
        
        config_helmholtz_energies = [
            config.helmholtz_energy.energy for config in self.configurations
        ]
        fk = config_helmholtz_energies
        kb = BOLTZMANN_CONSTANT
        t = self.configurations[0].helmholtz_energy.temperature
        
        
        z = np.sum(np.exp(
            -f/(kb*t)
        ))


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

