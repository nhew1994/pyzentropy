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
    intra = intra_entropy(configuration_probabilities, configuration_entropies)
    return gibbs + intra

def alt_system_entropy(
    partition_function,
    configuration_internal_energies,
    configuration_probabilities,
    temperature
):
    term_1 = -BOLTZMANN_CONSTANT*temperature*np.log(partition_function)
    term_2 = (1/temperature)*np.sum(
        configuration_probabilities*configuration_internal_energies
        )
    return term_1 + term_2

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
        configuration_helmholtz_energies,
        temperature
):
    pk = configuration_probabilities
    v = volumes
    fk = configuration_helmholtz_energies

    fk_spline = UnivariateSpline(v, fk, s=0, k=1)

    common_factor = (v/(BOLTZMANN_CONSTANT*temperature))
    term_1 = (np.sum(pk*fk_spline.derivative(n=1)(v)))**2
    term_2 = -(np.sum(pk*(fk_spline.derivative(n=1)(v))**2))

    return common_factor*(term_1 + term_2)

def system_bulk_modulus(
    system_helmholtz_energy,
    volumes
):
    fk_spline = UnivariateSpline(volumes, system_helmholtz_energy, s=0, k=2)
    return volumes*fk_spline.derivative(n=2)(volumes)

def alt_system_bulk_modulus(
    configuration_probabilities,
    volumes,
    configuration_helmholtz_energies,
    temperature
):
    intra = intra_bulk_modulus(
        configuration_probabilities,
        volumes,
        configuration_helmholtz_energies
    )
    inter = inter_bulk_modulus(
        configuration_probabilities,
        volumes,
        configuration_helmholtz_energies,
        temperature
    )
    return intra + inter

def configuration_internal_energy(
    configuration_free_energy,
    configuration_entropy,
    temperature
):
    return configuration_free_energy + temperature*configuration_entropy

def intra_heat_capacity(
    configuration_probabilities,
    configuration_heat_capacities,
):
    return np.sum(configuration_probabilities*configuration_heat_capacities)

def inter_heat_capcity(
    configuration_probabilities,
    configuration_internal_energies,
    temperature
):
    pk = configuration_probabilities
    uk = configuration_internal_energies

    common_factor = (1/(BOLTZMANN_CONSTANT*temperature**2))
    term_1 = np.sum(pk*uk**2)
    term_2 = -(np.sum(pk*uk))**2

    return common_factor*(term_1 + term_2)

def alt_heat_capacity(configuration_probabilities,
    configuration_heat_capacities,
    configuration_internal_energies,
    temperature
):
    intra = intra_heat_capacity(
        configuration_probabilities,
        configuration_heat_capacities
    )
    inter = inter_heat_capcity(
        configuration_probabilities,
        configuration_internal_energies,
        temperature
    )
    return intra + inter

def heat_capacity(system_entropy, temperature):
    s_spline = UnivariateSpline(temperature, system_entropy, s=0, k=1)
    
    return temperature*s_spline.derivative(n=1)(temperature)

