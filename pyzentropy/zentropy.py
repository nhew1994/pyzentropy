from scipy import constants
import numpy as np
from functools import cached_property
from scipy.interpolate import UnivariateSpline

BOLTZMANN_CONSTANT = constants.physical_constants[
    "Boltzmann constant in eV/K"
][0]

def heat_capacity(entropy, temperature):
    s_spline = UnivariateSpline(temperature, entropy, s=0, k=1)
    return temperature*s_spline.derivative(n=1)(temperature)

def gibbs_entropy(configuration_probabilities): # interconfigurational entropy
    return -BOLTZMANN_CONSTANT*np.sum(
        configuration_probabilities*np.log(configuration_probabilities)
        )

def intra_entropy(
    configuration_probabilities,
    configuration_entropies
): # just a suggestion for the name
    return np.sum(configuration_probabilities*configuration_entropies)

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
        name,
        structure,
        volume,
        temperature,
        internal_energy = None,
        entropy = None,
        helmholtz_energy = None,
    ):
        self.name = name
        self.structure = structure
        self.volume = volume
        self.temperature = temperature

        inputs = [internal_energy, entropy, helmholtz_energy]
        non_none_count = sum(arg is not None for arg in inputs)
        
        # Check if at least two arguments are provided
        if non_none_count < 2:
            raise ValueError("At least two of the following arguments must be "
                             "provided.: internal_energy, entropy, "
                             "helmholtz_energy")
        
        self.internal_energy = internal_energy
        self.entropy = entropy
        self.helmholtz_energy = helmholtz_energy

        if self.internal_energy is None:
            self.internal_energy = helmholtz_energy + temperature*entropy
        elif self.entropy is None:
            self.entropy = np.divide(
                internal_energy - helmholtz_energy,
                temperature,
                out=np.zeros_like(internal_energy),
                where=temperature!=0
            )
        elif self.helmholtz_energy is None:
            self.helmholtz_energy = internal_energy - temperature*entropy
        else:
            # validate consistency of the provided values
            pass

        self.partition_function = np.exp(
            -helmholtz_energy/(BOLTZMANN_CONSTANT*temperature)
        )

        self.heat_capacity = heat_capacity(self.entropy, self.temperature)
        self.probability = None


class System:
    def __init__(
        self,
        name,
        configurations: list
    ):
        self.name = name
        self.configurations = configurations

        same_volumes = self.same_configuration_volumes()
        if not same_volumes[0]:
            raise ValueError(
                "All configurations must have the same volume. "
                f"Configurations with mismatched volumes: {same_volumes[1]}"
            )
        self.volume = configurations[0].volume
        
        same_temperatures = self.same_configuration_temperatures()
        if not same_temperatures[0]:
            raise ValueError(
                "All configurations must have the same temperature. "
                f"Configurations with mismatched temperatures: {same_temperatures[1]}"
            )
        self.temperature = configurations[0].temperature

        self.partition_function = np.sum(
            [config.partition_function for config in configurations]
        )

        self.helmholtz_energy = -BOLTZMANN_CONSTANT*self.temperature*np.log(
            self.partition_function
        )

        pk = np.zeros(len(configurations))
        sk = np.zeros(len(configurations))
        cvk = np.zeros(len(configurations))
        uk = np.zeros(len(configurations))
        for i, config in enumerate(configurations):
            pk[i] = config.partition_function/self.partition_function
            sk[i] = config.entropy
            cvk[i] = config.heat_capacity
            uk[i] = config.internal_energy
        
        self.gibbs_entropy = gibbs_entropy(pk)
        self.inter_entropy = self.gibbs_entropy
        self.intra_entropy = np.sum(pk*sk)
        self.system_entropy = self.inter_entropy + self.intra_entropy
        
        f_spline = UnivariateSpline(
            self.volume, self.helmholtz_energy, s=0, k=2
        )
        self.bulk_modulus = self.volume * f_spline.derivative(n=2)(self.volume)
        
        s_spline = UnivariateSpline(self.temperature, sk, s=0, k=1)
        self.heat_capacity = self.temperature * s_spline.derivative(n=1)(self.temperature)


    def same_configuration_volumes(self):
        volumes = [config.volume for config in self.configurations]
        if not all(volume == volumes[0] for volume in volumes):
            mismatched_configs = [
                config for config in self.configurations if config.volume != volumes[0]
            ]
            return (False, mismatched_configs)
        return (True, None)

    def same_configuration_temperatures(self):
        temperatures = [config.temperature for config in self.configurations]
        if not all(temperature == temperatures[0] for temperature in temperatures):
            mismatched_configs = [
                config for config in self.configurations if config.temperature != temperatures[0]
            ]
            return (False, mismatched_configs)
        return (True, None)

    