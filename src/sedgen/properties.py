import re

# This could be done by constructing a complete periodic table database
# yourself from which necessary data could be extracted or by using an
# existing package (e.g. https://mendeleev.readthedocs.io/en/stable/)
# but would introduce, however, new requirements and perhaps
#  restrictions to other packages functionalities. Therefore, creating a
# restricted database with only the elements needed for the minerals
# with regard to SedGen could be the fastest short-term solution.


class Mineral:

    def __init__(self):

        self.molar_volume = 0
        self.molar_mass = 0
        self.mineral_formula = 0
        self.weathering_rate = 0

        self.mineral_strength = 0
        self.mineral_density = 0

        self.interface_strength = 0

        elements = {"Si": {"molar_volume": 0.000012054,
                           "density": 2.3290,
                           },
                    "O": {"molar_volume": 0.011196,
                          "density": 1.429,
                          }
                    }


class Quartz(Mineral):

    def __init__(self):
        Mineral.__init__(self)

        self.mineral_formula = "SiO2"
        self.molar_volume = 0
        self.mineral_density = 0
        self.molar_mass = 0

        self.weathering_rate = 0

        self.mineral_strength = 0

        self.interface_strength = 0


def get_elements(formula):
    elements = re.findall("([A-Z][a-z]?)([0-9]*)", formula)

    return elements


def calculate_molar_volume(mineral, weights):

    return sum((weights[e] * int(i)) for e, i in mineral)
