import numpy as np
from abc import ABC, abstractmethod
from scipy.interpolate import RegularGridInterpolator
from scipy.constants import physical_constants

BOLTZMANN_CONSTANT = physical_constants["Boltzmann constant in eV/K"][0]


class NDProperty(ABC):
    def __init__(self, variable_labels, property_label):
        self.variable_labels = variable_labels
        self.property_label = property_label
        self.dimensions = len(variable_labels)
    @abstractmethod
    def value(self, points):
        pass
    
    @abstractmethod
    def minimum(self):
        pass


class AnalyticalNDProperty(NDProperty):
    def __init__(self, function, variable_labels, property_label):
        super().__init__(variable_labels, property_label)
        self.function = function
    def value(self, points) -> float:
        return self.function(points)
    
    def minimum(self) -> float:
        raise NotImplementedError(
            "Logic to calculate the minimum of an analytical property has not "
            "been implemented"
        )
    



class TabulatedNDProperty(NDProperty):
    def __init__(
            self,
            variable_labels,
            points,
            values,
            method="linear",
            bounds_error=True,
            fill_value=np.nan,
            *, # enforce keyword-only arguments
            solver=None,
            solver_args=None
    ):
        super().__init__(variable_labels)
        self.interp = RegularGridInterpolator(
            points,
            values,
            method=method,
            bounds_error=bounds_error,
            fill_value=fill_value,
            solver=solver,
            solver_args=solver_args
        )
    def value(self, points):
        return self.interp(points)
    
    def minimum(self):
        raise NotImplementedError(
            "Logic to calculate the minimum of a tabulated property has not "
            "been implemented"
        )

    


class Configuration:
    def __init__(
            self,
            name,
            structure,
            multiplicity,
            internal_energy=None,
            helmholtz_energy=None,
            enthalpy=None,
            gibbs_energy=None,
            entropy=None,
            heat_capacity=None,
            bulk_modulus=None,
            thermal_expansion_coefficient=None
        ):
            self.name = name
            self.structure = structure
            self.multiplicity = multiplicity
            self._internal_energy = internal_energy
            self._helmholtz_energy = helmholtz_energy
            self._enthalpy = enthalpy
            self._gibbs_energy = gibbs_energy
            self._entropy = entropy
            self._heat_capacity = heat_capacity
            self._bulk_modulus = bulk_modulus
            self._thermal_expansion_coefficient = thermal_expansion_coefficient
    def __str__(self):
        return self.name
    
    @property
    def internal_energy(self):
        return self._internal_energy
        # elif self._variable_labels == ('T', 'S') or self._variable_labels == ('S', 'T'):
        #     if self._helmholtz_energy is not None and self._entropy is not None:
        #         return self._helmholtz_and_entropy_to_internal()
        # else:
        #     return None

    @internal_energy.setter
    def internal_energy(self, nd_property):
        self._internal_energy = nd_property

    @internal_energy.deleter
    def internal_energy(self):
        del self._internal_energy

    @property
    def helmholtz_energy(self):
        return self._helmholtz_energy
        # elif self._variable_labels == ('T', 'S') or self._variable_labels == ('S', 'T'):
        #     if self._internal_energy is not None and self._entropy is not None:
        #         return self._internal_and_entropy_to_helmholtz()

    
    @helmholtz_energy.setter
    def helmholtz_energy(self, nd_property):
        self._helmholtz_energy = nd_property

    @helmholtz_energy.deleter
    def helmholtz_energy(self):
        del self._helmholtz_energy
    
    @property
    def enthalpy(self):
        return self._enthalpy

    @enthalpy.setter
    def enthalpy(self, nd_property):
        self._enthalpy = nd_property
    
    @enthalpy.deleter
    def enthalpy(self):
        del self._enthalpy

    @property
    def gibbs_energy(self):
        return self._gibbs_energy
        
    @gibbs_energy.setter
    def gibbs_energy(self, nd_property):
        self._gibbs_energy = nd_property

    @gibbs_energy.deleter
    def gibbs_energy(self):
        del self._gibbs_energy
    
    @property
    def entropy(self):
        if self._entropy is not None:
            return self._entropy
        else:
            raise ValueError("Entropy must be provided")
        
    @entropy.setter
    def entropy(self, nd_property):
        self._entropy = nd_property
    
    @entropy.deleter
    def entropy(self):
        del self._entropy

    @property
    def heat_capcacity(self):
        return self._heat_capacity
        # elif self._entropy is not None:
        #     return self._entropy_to_heat_capacity()
        # else:
        #     return None
    
    @heat_capacity.setter
    def heat_capacity(self, nd_property):
        self._heat_capacity = nd_property

    @heat_capacity.deleter
    def heat_capacity(self):
        del self._heat_capacity

    @property
    def bulk_modulus(self):
        return self._bulk_modulus
    
    @bulk_modulus.setter
    def bulk_modulus(self, nd_property):
        self._bulk_modulus = nd_property

    @bulk_modulus.deleter
    def bulk_modulus(self):
        del self._bulk_modulus

    @property
    def thermal_expansion_coefficient(self):
        return self._thermal_expansion_coefficient
    
    @thermal_expansion_coefficient.setter
    def thermal_expansion_coefficient(self, nd_property):
        self._thermal_expansion_coefficient = nd_property

    @thermal_expansion_coefficient.deleter
    def thermal_expansion_coefficient(self):
        del self._thermal_expansion_coefficient
    
    def consistent_variables(self):
        # there may be problems with tuples and order e.g. ('S', 'T') vs ('T', 'S')
        # There may also be problems with later defing gibbs_energy as a function of T and P
        # from the Free energy, which is a function of T and V
        variable_labels = None
        for attribute_name in self.__dict__.keys():
            attribute = getattr(self, attribute_name)
            if isinstance(attribute, NDProperty):
                if variable_labels is None:
                    variable_labels = attribute.variable_labels
                elif attribute.variable_labels != self.variable_labels:
                    return False
        return True

    def variable_labels(self): # probably not needed
        if not self.consistent_variables():
            raise ValueError(
                "Variable labels are inconsistent among the properties of "
                "this configuration."
            )
        for attribute_name in self.__dict__.keys():
            attribute = getattr(self, attribute_name)
            if isinstance(attribute, NDProperty):
                return attribute.variable_labels
    

    # # Methods to derive properties from other properties
    # def _helmholtz_and_entropy_to_internal(self):
    #     raise NotImplementedError(
    #         "Logic to derive internal energy from Helmholtz energy and "
    #         "entropy has not been implemented"
    #     )
    # def _entropy_to_heat_capacity(self):
    #     raise NotImplementedError(
    #         "Logic to derive heat capacity from entropy has not been "
    #         "implemented"
    #     )
    # def _internal_and_entropy_to_helmholtz(self):
    #     raise NotImplementedError(
    #         "Logic to derive Helmholtz energy from internal energy and "
    #         "entropy has not been implemented"
    #     )


class HelmholtzConfiguration(Configuration):
    def __init__(
            self,
            name,
            structure,
            multiplicity,
            internal_energy=None,
            helmholtz_energy=None,
            entropy=None,
            heat_capacity=None,
            bulk_modulus=None,
            thermal_expansion_coefficient=None    
        ):
        super().__init__(
            name,
            structure,
            multiplicity,
            internal_energy,
            helmholtz_energy,
            entropy,
            heat_capacity,
            bulk_modulus,
            thermal_expansion_coefficient
        )
        self._variable_labels = ('T', 'V')

    @property
    def internal_energy(self):
        if self._internal_energy is not None:
            return self._internal_energy
        elif self._helmholtz_energy is not None and self._entropy is not None:
            return self._helmholtz_and_entropy_to_internal()
        else:
            return None
        
    @internal_energy.setter
    def internal_energy(self, nd_property):
        if not isinstance(nd_property, NDProperty):
            raise ValueError(
                "internal_energy must be an instance of the NDProperty class."
            )
        if nd_property.variable_labels != self._variable_labels:
            raise ValueError(
                "internal_energy must have variable labels ('T', 'V')"
            )
        self._internal_energy = nd_property

    @internal_energy.deleter
    def internal_energy(self):
        del self._internal_energy

    @property
    def helmholtz_energy(self):
        if self._helmholtz_energy is not None:
            return self._helmholtz_energy
        elif self._internal_energy is not None and self._entropy is not None:
            return self._internal_and_entropy_to_helmholtz()
        else:
            return None
    
    @helmholtz_energy.setter
    def helmholtz_energy(self, nd_property):
        if not isinstance(nd_property, NDProperty):
            raise ValueError(
                "helmholtz_energy must be an instance of the NDProperty class."
            )
        if nd_property.variable_labels != self._variable_labels:
            raise ValueError(
                "helmholtz_energy must have variable labels ('T', 'V')"
            )
        self._helmholtz_energy = nd_property

    @helmholtz_energy.deleter
    def helmholtz_energy(self):
        del self._helmholtz_energy

    @property
    def enthalpy(self):
        pass

    @property
    def gibbs_energy(self):
        pass

    @property
    def entropy(self):
        if self._entropy is not None:
            return self._entropy
        elif self._internal_energy is not None and self._helmholtz_energy is not None:
            return self._internal_and_helmholtz_to_entropy()
        else:
            return None
        
    @entropy.setter
    def entropy(self, nd_property):
        if not isinstance(nd_property, NDProperty):
            raise ValueError(
                "entropy must be an instance of the NDProperty class."
            )
        if nd_property.variable_labels != self._variable_labels:
            raise ValueError(
                "entropy must have variable labels ('T', 'V')"
            )
        self._entropy = nd_property

    @entropy.deleter
    def entropy(self):
        del self._entropy

    @property
    def heat_capacity(self):
        if self._heat_capacity is not None:
            return self._heat_capacity
        elif self._entropy is not None:
            return self._entropy_to_heat_capacity()
        else:
            return None
        
    @heat_capacity.setter
    def heat_capacity(self, nd_property):
        if not isinstance(nd_property, NDProperty):
            raise ValueError(
                "heat_capacity must be an instance of the NDProperty class."
            )
        if nd_property.variable_labels != self._variable_labels:
            raise ValueError(
                "heat_capacity must have variable labels ('T', 'V')"
            )
        self._heat_capacity = nd_property

    @heat_capacity.deleter
    def heat_capacity(self):
        del self._heat_capacity


                                 
            


class EnthalpyConfiguration(Configuration):
    pass
class GibbsConfiguration(Configuration):
    pass
class InternalConfiguration(Configuration):
    pass


class System:
    def __init__(self, name, configurations=()):
        """
        Initialize a System object.

        Args:
            name (str): Name of the system.
            configurations (tuple): Iterable of Configuration objects. Defaults to an empty tuple.

        Raises:
            ValueError: If any item in configurations is not a Configuration instance,
                        or if duplicate configuration names are found.
        """
        self.name = name
        self.configurations = {}
        self._probabilities = None # dictionary of probabilities
        self._internal_energy = None
        self._helmholtz_energy = None
        self._enthalpy = None
        self._gibbs_energy = None
        self._entropy = None
        self._heat_capacity = None
        self._bulk_modulus = None
        self._thermal_expansion_coefficient = None

        for i, configuration in enumerate(configurations):
            if not isinstance(configuration, Configuration):
                raise ValueError(
                    "All configurations must be instances of the "
                    f"Configuration class. The object {configuration} at index "
                    f"{i} is not an instance of the Configuration class."
                )
            if configuration.name in self.configurations:
                raise ValueError(
                    f"All configurations must have unique names. The name "
                    f"'{configuration.name}' appeared more than once."
                )
            self.configurations[configuration.name] = configuration
   
    @property
    def probabilities(self):
        return self._probabilities
    
    @probabilities.setter
    def probabilities(self, probabilities):
        self._probabilities = probabilities

    @probabilities.deleter
    def probabilities(self):
        del self._probabilities

    @property
    def internal_energy(self):
        return self._internal_energy

    @internal_energy.setter
    def internal_energy(self, nd_property):
        if not isinstance(nd_property, NDProperty):
            raise ValueError(
                "internal_energy must be an instance of the NDProperty class."
            )
        self._internal_energy = nd_property   

    @internal_energy.deleter
    def internal_energy(self):
        del self._internal_energy

    @property
    def helmholtz_energy(self):
        return self._helmholtz_energy
    
    @helmholtz_energy.setter
    def helmholtz_energy(self, nd_property):
        if not isinstance(nd_property, NDProperty):
            raise ValueError(
                "helmholtz_energy must be an instance of the NDProperty class."
            )
        self._helmholtz_energy = nd_property

    @helmholtz_energy.deleter
    def helmholtz_energy(self):
        del self._helmholtz_energy

    @property
    def enthalpy(self, nd_property):
        if not isinstance(nd_property, NDProperty):
            raise ValueError(
                "enthalpy must be an instance of the NDProperty class."
            )
        self._enthalpy = nd_property
        
    @enthalpy.deleter
    def enthalpy(self):
        del self._enthalpy

    @property
    def gibbs_energy(self, nd_property):
        if not isinstance(nd_property, NDProperty):
            raise ValueError(
                "gibbs_energy must be an instance of the NDProperty class."
            )
        self._gibbs_energy = nd_property

    @gibbs_energy.deleter
    def gibbs_energy(self):
        del self._gibbs_energy

    @property
    def entropy(self, nd_property):
        if not isinstance(nd_property, NDProperty):
            raise ValueError(
                "entropy must be an instance of the NDProperty class."
            )
        self._entropy = nd_property
        
    @entropy.deleter
    def entropy(self):
        del self._entropy

    @property
    def heat_capacity(self, nd_property):
        if not isinstance(nd_property, NDProperty):
            raise ValueError(
                "heat_capacity must be an instance of the NDProperty class."
            )
        self._heat_capacity = nd_property

    @heat_capacity.deleter
    def heat_capacity(self):
        del self._heat_capacity

    @property
    def bulk_modulus(self, nd_property):
        if not isinstance(nd_property, NDProperty):
            raise ValueError(
                "bulk_modulus must be an instance of the NDProperty class."
            )
        self._bulk_modulus = nd_property

    @bulk_modulus.deleter
    def bulk_modulus(self):
        del self._bulk_modulus

    @property
    def thermal_expansion_coefficient(self, nd_property):
        if not isinstance(nd_property, NDProperty):
            raise ValueError(
                "thermal_expansion_coefficient must be an instance of the NDProperty class."
            )
        self._thermal_expansion_coefficient = nd_property

    @thermal_expansion_coefficient.deleter
    def thermal_expansion_coefficient(self):
        del self._thermal_expansion_coefficient
    

# This might be appropriate for the HelmholtzSystem class
    # @property
    # def helmholtz_energy(self, points): # relative to the lowest energy configuration
    #     # Check that all configurations have a Helmholtz energy
    #     for configuration in self.configurations:
    #         if configuration.helmholtz_energy is None:
    #             raise ValueError("Helmholtz energy must be provided for all "
    #                              "configurations"
    #             )
    #     # calculate the system's Helmholtz energy
    #     for configuration in self.configurations:

    #         if configuration.helmholtz_energy.value(points) == min(
    #                 configuration.helmholtz_energy.value(points)
    #         ):
                
class HelmholtzSystem(System):
    def __init__(self, name, configurations=()):
        super().__init__(name, configurations)
        self._variable_labels = ('T', 'V')

    @property
    def check_configurations_for_property(self, property):
        # might make this more complex later to tell user which configurations
        # are missing the property. might want a simple one just to get the
        # boolean value, and a more complex one to get which configurations.
        for configuration in self.configurations:
            if getattr(configuration, property) is None:
                return False
        return True

    def _helmholtz_k_to_probabilities(self, points): # change to points?
        """
        Compute the probabilities of each configuration in the system based on
        the Helmholtz energies of the configurations. done in a numerically
        stable way using a log-sum-exp approach.
        """
        all_ln_zk = np.zeros((
            len(self.configurations),
            len(points[0]),
            len(points[1])
        )) # check if points[0] and points[1] are in the correct spot
        configuration_names = []
        for i, configuration in enumerate(self.configurations):
            if configuration.helmholtz_energy is None:
                raise ValueError(
                    "Helmholtz energy must be provided for all configurations"
                )
            wk = configuration.multiplicity
            fk = configuration.helmholtz_energy.value(points)
            t = points[0] # temperature
            kb = BOLTZMANN_CONSTANT
            ln_zk = np.log(wk) - fk/(kb*t) # check if t is column or row vector
            all_ln_zk[i] = ln_zk
            # making the list this way to make sure order is preserved.
            # not sure if it is necessary.
            configuration_names.append(configuration.name)
        ln_zk_max = np.max(all_ln_zk, axis=0)
        np_probabilities = np.exp(all_ln_zk - ln_zk_max)/np.sum(ln_zk_max, axis=0)
        # make np_probabilities into NDProperty objects
        variable_labels = ('T', 'V')
        property_label = 'probability' # might want to change this
        probabilities = {}
        for i, pk in enumerate(np_probabilities):
            values = pk
            nd_pk = TabulatedNDProperty(variable_labels, points, values)
            probabilities[configuration_names[i]] = nd_pk
        return probabilities

    @property
    def probabilities(self):
        if self._probabilities is not None:
            return self._probabilities
        elif self.check_configurations_for_property('helmholtz_energy'):
            return self._helmholtz_to_probabilities()
        else:
            return None

    @probabilities.setter
    def probabilities(self, probabilities):
        self._probabilities = probabilities

    @probabilities.deleter
    def probabilities(self):
        del self._probabilities

    @property
    def internal_energy(self):
        if self._internal_energy is not None:
            return self._internal_energy
        elif self._helmholtz_energy is not None and self._entropy is not None:
            return self._helmholtz_and_entropy_to_internal()
        else:
            return None
        
    @internal_energy.setter
    def internal_energy(self, nd_property):
        if not isinstance(nd_property, NDProperty):
            raise ValueError(
                "internal_energy must be an instance of the NDProperty class."
            )
        if nd_property.variable_labels != self._variable_labels:
            raise ValueError(
                "internal_energy must have variable labels ('T', 'V')"
            )
        self._internal_energy = nd_property

    @internal_energy.deleter
    def internal_energy(self):
        del self._internal_energy

    @property
    def helmholtz_energy(self):
        if self._helmholtz_energy is not None:
            return self._helmholtz_energy
        elif self._internal_energy is not None and self._entropy is not None:
            return self._internal_and_entropy_to_helmholtz()
        elif self.check_configurations_for_property('helmholtz_energy'):
            return self._helmholtz_k_to_probabilities()
        else:
            return None
    
    @helmholtz_energy.setter
    def helmholtz_energy(self, nd_property):
        if not isinstance(nd_property, NDProperty):
            raise ValueError(
                "helmholtz_energy must be an instance of the NDProperty class."
            )
        if nd_property.variable_labels != self._variable_labels:
            raise ValueError(
                "helmholtz_energy must have variable labels ('T', 'V')"
            )
        self._helmholtz_energy = nd_property

    @helmholtz_energy.deleter
    def helmholtz_energy(self):
        del self._helmholtz_energy

    @property
    def enthalpy(self):
        pass

    @property
    def gibbs_energy(self):
        pass

    @property
    def entropy(self):
        if self._entropy is not None:
            return self._entropy
        elif self._internal_energy is not None and self._helmholtz_energy is not None:
            return self._internal_and_helmholtz_to_entropy()
        elif self.check_configurations_for_property('entropy'):
            return 
        else:
            return None
    
    @entropy.setter
    def entropy(self, nd_property):
        if not isinstance(nd_property, NDProperty):
            raise ValueError(
                "entropy must be an instance of the NDProperty class."
            )
        if nd_property.variable_labels != self._variable_labels:
            raise ValueError(
                "entropy must have variable labels ('T', 'V')"
            )
        self._entropy = nd_property

    @entropy.deleter
    def entropy(self):
        del self._entropy
    
    @property
    def heat_capacity(self):
        if self._heat_capacity is not None:
            return self._heat_capacity
        elif self._entropy is not None:
            return self._entropy_to_heat_capacity()
        else:
            return None
        
    @heat_capacity.setter
    def heat_capacity(self, nd_property):
        if not isinstance(nd_property, NDProperty):
            raise ValueError(
                "heat_capacity must be an instance of the NDProperty class."
            )
        if nd_property.variable_labels != self._variable_labels:
            raise ValueError(
                "heat_capacity must have variable labels ('T', 'V')"
            )
        self._heat_capacity = nd_property

    @heat_capacity.deleter
    def heat_capacity(self):
        del self._heat_capacity
