from scipy import constants
import numpy as np
from scipy.interpolate import UnivariateSpline

BOLTZMANN_CONSTANT = constants.physical_constants[
    "Boltzmann constant in eV/K"
][0]


def canonical_partition_function(
    config_helmholtz_energeries: np.array,
    temperature
):
    z_k = np.exp(
        -config_helmholtz_energeries /( BOLTZMANN_CONSTANT*temperature)
        )
    return np.sum(z_k)

def system_helmholtz_energy(
    partition_function: float,
    temperature: float
):
    return -BOLTZMANN_CONSTANT*temperature*np.log(partition_function)

def configuration_probabilities(
    config_helmholtz_energies,
    system_helmholtz_energy,
    temperature
):
    p = np.exp(
        -(config_helmholtz_energies-system_helmholtz_energy)/(
            BOLTZMANN_CONSTANT*temperature
        )
    )
    return p

def gibbs_entropy(configuration_probabilities): # interconfigurational entropy
    return -BOLTZMANN_CONSTANT*np.sum(
        configuration_probabilities*np.log(configuration_probabilities)
        )
    
def intra_entropy(
    configuration_probabilities,
    configuration_entropies
): # just a suggestion for the name
    return np.sum(configuration_probabilities*configuration_entropies)

def system_entropy(configuration_probabilities, configuration_entropies):
    gibbs = gibbs_entropy(configuration_probabilities)
    intra = intraconfigurational_entropy(
        configuration_probabilities,
        configuration_entropies
    )
    return gibbs + intra

def configuration_entropies(
    config_helmholtz_energies,
    system_helmholtz_energy,
    temperature
):
    configuration_probabilities = configuration_probabilities(
        config_helmholtz_energies,
        system_helmholtz_energy,
        temperature
    )
    summation = np.sum(configuration_probabilities*config_helmholtz_energies)
    return (summation-system_helmholtz_energy)/temperature

def intra_bulk_modulus(
        configuration_probabilities,
        volumes,
        configuration_helmholtz_energies
):
    pk = configuration_probabilities
    v = volumes
    fk = configuration_helmholtz_energies

    fk_spline = UnivariateSpline(v, fk, s=0, k=2)
    return np.sum(pk*v*fk_spline.derivative(n=2)(v))

def inter_bulk_modulus(
        configuration_probabilities,
        volumes,
        configuration_helmholtz_energies
):
    pk = configuration_probabilities
    v = volumes
    fk = configuration_helmholtz_energies

    fk_spline = UnivariateSpline(v, fk, s=0, k=2)
    return np.sum(pk*fk_spline.derivative(n=2)(v))

