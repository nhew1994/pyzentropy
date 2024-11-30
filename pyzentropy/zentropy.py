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
        configuration_probabilities*np.log(configuration_probabilities),
        axis=0
        )

def intra_entropy(
    configuration_probabilities,
    configuration_entropies
): # just a suggestion for the name
    return np.sum(configuration_probabilities*configuration_entropies)


class Configuration:
    def __init__(
        self,
        name,
        structure,
        volume,
        temperature,
        internal_energy = None, # should generalize to internal energy
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
            self.internal_energy = helmholtz_energy + temperature[:, np.newaxis]*entropy
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
            -helmholtz_energy/(BOLTZMANN_CONSTANT*temperature[:, np.newaxis])
        )

        heat_caps = np.zeros(entropy.shape)
        for i in range(entropy.shape[1]):
            s_spline = UnivariateSpline(temperature, entropy[:, i], s=0, k=1)
            heat_caps[:, i] = temperature.squeeze() * s_spline.derivative(n=1)(temperature.squeeze()) # fix temperature problems
        self.heat_capacity = heat_caps
        self.probability = None


class System:
    def __init__(
        self,
        name,
        configurations: list
    ):
        self.name = name
        self.configurations = configurations

        v_and_t = ("volume", "temperature")
        for attribute in v_and_t:
            mismatched = self.mismatched_attribute(attribute)
            if mismatched[0]:
                raise ValueError(
                    f"All configurations must have the same {attribute}. "
                    f"Configurations with mismatched {attribute}s: {mismatched[1]}")
        self.volume = configurations[0].volume
        self.temperature = configurations[0].temperature

        self.partition_function = np.sum(
            [config.partition_function for config in configurations],axis=0
        )

        self.helmholtz_energy = -BOLTZMANN_CONSTANT*self.temperature[:, np.newaxis]*np.log(
            self.partition_function
        )

        pk_list = [
            config.partition_function / self.partition_function
            for config in configurations
        ]
        sk_list = [config.entropy for config in configurations]
        cvk_list = [config.heat_capacity for config in configurations]
        uk_list = [config.internal_energy for config in configurations]
        pk = np.array(pk_list)
        sk = np.array(sk_list)
        cvk = np.array(cvk_list)
        uk = np.array(uk_list)
        
        self.gibbs_entropy = gibbs_entropy(pk)
        self.inter_entropy = self.gibbs_entropy
        self.intra_entropy = np.sum(pk*sk,axis=0)
        self.entropy = self.inter_entropy + self.intra_entropy
        
        b = np.zeros(self.helmholtz_energy.shape)
        for i in range(self.helmholtz_energy.shape[1]):
            
            f_spline = UnivariateSpline(
                self.volume, self.helmholtz_energy[i,:], s=0, k=2
            )
            b[i,:] = self.volume * f_spline.derivative(n=2)(self.volume)
        self.bulk_modulus = b
        
        heat_caps = np.zeros(self.entropy.shape)
        for i in range(self.entropy.shape[1]):
            s_spline = UnivariateSpline(self.temperature, self.entropy[:, i], s=0, k=1)
            heat_caps[:, i] = self.temperature * s_spline.derivative(n=1)(self.temperature)
        self.heat_capacity = heat_caps

    def mismatched_attribute(self, attribute):
        reference_value = getattr(self.configurations[0], attribute)
        mismatched_configs = []
        for config in self.configurations:
            current_value = getattr(config, attribute)
            matching = np.array_equal(current_value, reference_value)
            if not matching:
                mismatched_configs.append(config.name)
        if mismatched_configs:
            return (True, mismatched_configs)
        return (False, None)
    
    