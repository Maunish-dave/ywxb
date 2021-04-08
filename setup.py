from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'Neural Network using only numpy'
LONG_DESCRIPTION = 'A package that allows to build neural network using only numpy'

# Setting up
setup(
    name="ywxb",
    version=VERSION,
    author="Maunish Dave",
    author_email="maunish1009@gmail.com",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=['numpy'],
    keywords=['python','numpy'],
)