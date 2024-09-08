from setuptools import find_packages, setup
from typing import List


hypen_e_dot = "-e ."
def get_requirements(file_path:str)->List[str]:
    requirements=[]
    with open(file_path) as file:
        requirements = file.readlines()
        requirements = [req.replace("\n"," ") for req in requirements]

        if hypen_e_dot in requirements:
            requirements.remove(hypen_e_dot)
    return requirements


setup(
name = 'ML Project',
version = '0.0.1',
author = 'Abani',
author_email = 'abanis24@gmail.com',
packages = find_packages(),
install_requires = get_requirements('requirements.txt')
)