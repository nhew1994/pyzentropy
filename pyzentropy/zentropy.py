import numpy as np
from abc import ABC, abstractmethod
from scipy.interpolate import RegularGridInterpolator
from scipy.constants import physical_constants
from scipy.interpolate import interp1d

BOLTZMANN_CONSTANT = physical_constants["Boltzmann constant in eV/K"][0]

class NDProperty(ABC):
    def __init__(self, variable_labels, property_label):
        self.variable_labels = variable_labels
        self.property_label = property_label
        self.dimensions = len(variable_labels)

    def __str__(self):
        return f"{self.property_label}({', '.join(self.variable_labels)})"
    @abstractmethod
    def calculate_value(self, points):
        pass
    
    @abstractmethod
    def minimize(self):
        pass

# add these methods in the future
#     def __add__(self, other):
#         pass

#     def __sub__(self, other):
#         pass

# +	__add__(self, other)	obj1 + obj2
# -	__sub__(self, other)	obj1 - obj2
# *	__mul__(self, other)	obj1 * obj2
# /	__truediv__(self, other)	obj1 / obj2
# //	__floordiv__(self, other)	obj1 // obj2
# %	__mod__(self, other)	obj1 % obj2
# **	__pow__(self, other)	obj1 ** obj2
# ==	__eq__(self, other)	obj1 == obj2
# !=	__ne__(self, other)	obj1 != obj2
# <	__lt__(self, other)	obj1 < obj2
# <=	__le__(self, other)	obj1 <= obj2
# >	__gt__(self, other)	obj1 > obj2
# >=	__ge__(self, other)	obj1 >= obj2

class AnalyticalNDProperty(NDProperty):
    def __init__(self, function, variable_labels, property_label):
        super().__init__(variable_labels, property_label)
        self.function = function
    def calculate_value(self, points) -> float:
        return self.function(points)
    
    def minimize(self) -> float:
        raise NotImplementedError(
            "Logic to calculate the minimum of an analytical property has not "
            "been implemented"
        )
    


class TabulatedNDProperty(NDProperty):
    def __init__(
            self,
            variable_labels,
            property_label,
            points,
            values,
            method="linear",
            bounds_error=True,
            fill_value=np.nan,
            *, # enforce keyword-only arguments
            solver=None,
            solver_args=None
    ):
        super().__init__(variable_labels, property_label)
        self.interp = RegularGridInterpolator(
            points,
            values,
            method=method,
            bounds_error=bounds_error,
            fill_value=fill_value,
            solver=solver,
            solver_args=solver_args
        )
        self.points = points
        self.values = values

    def calculate_value(self, points):
        return self.interp(points)
    
    def minimize(self):
        raise NotImplementedError(
            "Logic to calculate the minimum of a tabulated property has not "
            "been implemented"
        )
    def partial_derivative(
            self,
            wrt_index, # change to 'T' or 'V' etc in the future perhaps
            new_label,
            scaling_factor=1 # temporary till NDProperty multiplication is implemented
    ):
        """
        Compute the partial derivative of the property with respect to one of
        its variables.

        Args:
            wrt_index (int): Index of the variable with respect to which the
                             derivative is taken.
            new_label (str): Label of the new variable.

        Returns:
            TabulatedNDProperty: A new TabulatedNDProperty object representing
                                 the partial derivative of the property.
        """
        if new_label is None:
            new_label = f"d({self.property_label})/d({self.variable_labels[wrt_index]})"

        coord = self.points[wrt_index]

        derivative_array = np.gradient(
            self.values,
            coord,      # spacing along the axis in question
            axis=wrt_index
        )

        derivative_array *= scaling_factor

        return TabulatedNDProperty(
            variable_labels=self.variable_labels,
            property_label=new_label,
            points=self.points,
            values=derivative_array,
            method=self.interp.method,
            bounds_error=self.interp.bounds_error,
            fill_value=self.interp.fill_value
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
    def heat_capacity(self):
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


class NDConfiguration(Configuration):
    def __init__(
            self,
            name,
            structure,
            variable_labels,
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
        self._variable_labels = variable_labels

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
                "internal_energy must have variable labels "
                f"{self._variable_labels}"
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
                "helmholtz_energy must have variable labels "
                f"{self._variable_labels}"
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
                "entropy must have variable labels "
                f"{self._variable_labels}"
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
                "heat_capacity must have variable labels "
                f"{self._variable_labels}"
            )
        self._heat_capacity = nd_property

    @heat_capacity.deleter
    def heat_capacity(self):
        del self._heat_capacity


        

    

            


class EnthalpyConfiguration(Configuration):
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
        self._pressure = None

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
    
    def check_configurations_for_property(self, property):
        # might make this more complex later to tell user which configurations
        # are missing the property. might want a simple one just to get the
        # boolean value, and a more complex one to get which configurations.
        for name, configuration in self.configurations.items():
            if getattr(configuration, property) is None:
                return False
        return True

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
    

# This might be appropriate for the NVTSystem class
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
                
class NVTSystem(System):
    def __init__(self, name, configurations=()):
        super().__init__(name, configurations)
        self._variable_labels = ('T', 'V')

    def _helmholtz_k_to_probabilities(self):
        """
        Compute the probabilities of each configuration in the system based on
        the Helmholtz energies of the configurations. done in a numerically
        stable way using a log-sum-exp approach.
        """
        first_config = next(iter(self.configurations.items()))
        points = first_config[1].helmholtz_energy.points
        temperature = points[0]
        volume = points[1]
        all_ln_zk = np.zeros((
            len(self.configurations),
            len(temperature),
            len(volume)
        ))
        names = []
        for k, (name, configuration) in enumerate(self.configurations.items()):
            names.append(name) # used below to make a dictionary
            if configuration.helmholtz_energy is None:
                raise ValueError(
                    "Helmholtz energy must be provided for all configurations"
                )
            wk = configuration.multiplicity
            fk = configuration.helmholtz_energy.values
            t = configuration.helmholtz_energy.points[0] # temperature
            t = t[:, np.newaxis] # make t a column vector for the division
            kb = BOLTZMANN_CONSTANT

            ln_zk = np.log(wk) - fk/(kb*t)
            all_ln_zk[k] = ln_zk
        ln_zk_max = np.max(all_ln_zk, axis=0)
        exps = np.exp(all_ln_zk - ln_zk_max) # Z_k
        denominator = np.sum(exps, axis=0) # Z
        probabilities = exps/denominator # Z_k/Z
        # make probabilities an NDProperty object
        variable_labels = ('T', 'V')
        property_label = 'p' # might want to change this

        probabilities_dict = {}
        for k, prob in enumerate(probabilities):
            values = prob
            nd_prob = TabulatedNDProperty(
                variable_labels,
                property_label,
                points,
                values
            )
            name = names[k]
            probabilities_dict[name] = nd_prob
        return probabilities_dict
    
    def _helmholtz_k_and_probability_k_to_helmholtz(self):
        """
        Compute the Helmholtz energy of the system based on the Helmholtz
        energies of the configurations and the probabilities of each
        configuration.
        """
        first_config = next(iter(self.configurations.items()))
        points = first_config[1].helmholtz_energy.points
        temperature = points[0]
        volume = points[1]
        kb = BOLTZMANN_CONSTANT
        fk_pk_array = np.zeros((
            len(self.configurations),
            len(temperature),
            len(volume)
        ))
        pk_ln_pk_array = np.zeros((
            len(self.configurations),
            len(temperature),
            len(volume)
        ))
        for i, (name, config) in enumerate(self.configurations.items()):
            if config.helmholtz_energy is None:
                raise ValueError(
                    "Helmholtz energy must be provided for all configurations"
                )
            if self.probabilities[name] is None:
                raise ValueError(
                    "Probabilities must be provided for all configurations"
                )
            fk = config.helmholtz_energy.values 
            pk = self.probabilities[name].values
            fk_pk_array[i] = fk * pk
            pk_ln_pk_array[i] = np.where(pk > 0.0, pk * np.log(pk), 0.0)
        intra_helmholtz = np.sum(fk_pk_array, axis=0)
        t = temperature[:, np.newaxis] # make t a column vector
        inter_helmholtz = kb * t * np.sum(pk_ln_pk_array, axis=0)
        helmholtz = intra_helmholtz + inter_helmholtz
        # make helmholtz an NDProperty object
        variable_labels = ('T', 'V')
        property_label = 'F'
        helmholtz_nd = TabulatedNDProperty(
            variable_labels,
            property_label,
            points,
            helmholtz
        )
        return helmholtz_nd
    
    def _helmholtz_to_pressure(self):
        """
        Compute the pressure of the system based on the Helmholtz energy of the
        system.
        """
        f = self.helmholtz_energy
        points = f.points
        volume = points[1]
        pressure = -np.gradient(f.values, volume, axis=1)
        # return pressure as an NDProperty object
        variable_labels = ('T', 'V')
        property_label = 'P'
        pressure_nd = TabulatedNDProperty(
            variable_labels,
            property_label,
            points,
            pressure
        )
        return pressure_nd

    # def _helmholtz_to_gibbs(self):
    #     """
    #     Compute the Gibbs energy of the system based on the Helmholtz energy
    #     of the system.
    #     """
    #     f = self.helmholtz_energy
    #     temperature = f.points[0]
    #     volume = f.points[1]
    #     kb = BOLTZMANN_CONSTANT
    #     pressure = 
    #     g = f.values + pressure*volume

        
    @property
    def probabilities(self):
        if self._probabilities is not None:
            return self._probabilities
        elif self.check_configurations_for_property('helmholtz_energy'):
            self._probabilities = self._helmholtz_k_to_probabilities()
            return self._helmholtz_k_to_probabilities()
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
            self._helmholtz_energy = self._helmholtz_k_and_probability_k_to_helmholtz()
            return self._helmholtz_energy
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


    @property
    def gibbs_energy(self):
        if self._gibbs_energy is not None:
            return self._gibbs_energy
        elif self._helmholtz_energy is not None:
            return self._helmholtz_to_gibbs()
        else:
            return None

    @property 
    def pressure(self):
        if self._pressure is not None:
            return self._pressure
        elif self._helmholtz_energy is not None:
            return self._helmholtz_to_pressure()
        else:
            return None
        
def build_f_of_tp(F_TV, P_TV, new_T_points, new_P_points): #likely temporary function
    """
    Not 100% sure this is correct. the result looks good.
    """
    # Unpack data from the F(T,V) property
    T_points, V_points = F_TV.points     # e.g. (array_of_T, array_of_V)
    F_values = F_TV.values              # shape (nT, nV)
    # Unpack data from the P(T,V) property
    P_values = P_TV.values              # shape (nT, nV)

    # We will create a 2D array of size (len(new_T_points), len(new_P_points))
    F_TP_values = np.zeros((len(new_T_points), len(new_P_points)), dtype=float)
    F_TP_values.fill(np.nan)  # initialize with NaN

    # For fast lookups, make 2D interpolators (optional if you already have them)
    F_interp = RegularGridInterpolator((T_points, V_points), F_values, bounds_error=False, fill_value=np.nan)
    P_interp = RegularGridInterpolator((T_points, V_points), P_values, bounds_error=False, fill_value=np.nan)

    # For each T_i in new_T_points:
    for i, T_i in enumerate(new_T_points):
        # 1) Extract the 1D slice of P vs V at that T:
        #    We'll do this by using P_interp at [T_i, V_points].
        #    If T_i is exactly one of the T_points, we can index directly.
        #    Otherwise, we can use an interpolator to get the row.

        # Let's do a direct interpolation approach:
        P_slice = P_interp(np.array([[T_i, v] for v in V_points]))  # shape = (nV,)

        # Now build a 1D interpolation: P_slice(v) -> v
        # Make sure there's at least one strictly monotonic interval.
        # If P_slice is decreasing in v, invert by flipping v_points and P_slice.
        # We'll assume it's strictly decreasing in V, so P_slice[0] > P_slice[-1].
        # If it's the opposite, we can just reorder them.

        # Check monotonic direction:
        if P_slice[0] < P_slice[-1]:
            # It's increasing in V, so let's invert that interpolation
            v_points_for_interp = V_points
            P_points_for_interp = P_slice
        else:
            # It's decreasing, so reverse them for a monotonic ascending function
            v_points_for_interp = V_points[::-1]
            P_points_for_interp = P_slice[::-1]

        # Create an interpolator from P -> V
        #   so that if we input a particular P, we get a V that yields that P
        P_to_V = interp1d(
            P_points_for_interp,
            v_points_for_interp,
            kind='linear',
            bounds_error=False,
            fill_value=np.nan
        )

        # 2) For each new_P in new_P_points, find the corresponding volume V_i
        for j, P_j in enumerate(new_P_points):
            # If this P_j is outside the range of P_slice, we'll get NaN
            V_solution = P_to_V(P_j)

            # 3) Evaluate F(T_i, V_solution)
            if not np.isnan(V_solution):
                F_TP_values[i, j] = F_interp([T_i, V_solution])
            else:
                F_TP_values[i, j] = np.nan  # out of range

    # Finally, build a new TabulatedNDProperty for F(T,P)
    F_TP = TabulatedNDProperty(
        variable_labels=("T", "P"),
        property_label="F",
        points=(new_T_points, new_P_points),
        values=F_TP_values,
        method="linear",
        bounds_error=False,
        fill_value=np.nan
    )
    return F_TP