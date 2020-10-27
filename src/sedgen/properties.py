import re

# This could be done by constructing a complete periodic table database
# yourself from which necessary data could be extracted or by using an
# existing package (e.g. https://mendeleev.readthedocs.io/en/stable/)
# but would introduce, however, new requirements and perhaps
#  restrictions to other packages functionalities. Therefore, creating a
# restricted database with only the elements needed for the minerals
# with regard to SedGen could be the fastest short-term solution.


class Mineral:

    elements = {"Si": {"molar_volume": 12.054,
                       "density": 2.3290,
                       },
                "O": {"molar_volume": 0,
                      "density": 1.429,
                      },
                }

    def get_elements(self):
        elements = re.findall("([A-Z][a-z]?)([0-9]*)", self.formula)

        return elements


class Quartz(Mineral):

    def __init__(self):
        self.formula = "SiO2"
        self.molar_volume = 23.69  # cm³/mol
        self.density = 2.66  # g/cm³ | calculated
        self.molar_mass = 60.09  # g

        self.chem_weath_rate = 0
        self.strength = 0
        # self.interface_strength = 0


class Plagioclase(Mineral):
    def __init__(self):
        pass


class Kfeldspar(Mineral):
    def __init__(self):
        pass


class Biotite(Mineral):
    def __init__(self):
        pass


class Opaques(Mineral):
    """Use properties of magnetite here or mean values of some common
    opaque minerals the occur in granite?"""
    def __init__(self):
        pass


class Accessories(Mineral):
    """Use properties of hornblende here or mean values of some common
    accessory minerals that occur in granite?"""
    def __init__(self):
        pass


# def calculate_molar_volume(mineral, weights):

#     return sum((weights[e] * int(i)) for e, i in mineral)
