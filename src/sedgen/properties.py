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

        self.molar_volume =
        self.molar_mass =
        self.mineral_formula =
        self.weathering_rate =

        self.mineral_strength =
        self.mineral_density =

        self.interface_strength =

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
        self.molar_volume =
        self.mineral_density =
        self.molar_mass =

        self.weathering_rate =

        self.mineral_strength =

        self.interface_strength =


def get_elements(formula):
    elements = re.findall("([A-Z][a-z]?)([0-9]*)", formula)

    return elements


def calculate_molar_volume(mineral):

    return sum((weights[e] * int(i)) for e, i in mineral)
