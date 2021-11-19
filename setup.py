""" Setup script for cdi package. """

from setuptools import setup
from os import path

# here = path.abspath(path.dirname(__file__))
# # Get the long description from the README file
# with open(path.join(here, 'README.md'), encoding='utf-8') as f:
#     long_description = f.read()

long_description = ('Statistical models are central to machine learning with broad applicability across a range of downstream tasks. The models are typically controlled by free parameters that are estimated from data by maximum-likelihood estimation. However, when faced with real-world datasets many of the models run into a critical issue: they are formulated in terms of fully-observed data, whereas in practice the datasets are plagued with missing data. The theory of statistical model estimation from incomplete data is conceptually similar to the estimation of latent-variable models, where powerful tools such as variational inference (VI) exist. However, in contrast to standard latent-variable models, parameter estimation with incomplete data often requires estimating exponentially-many conditional distributions of the missing variables, hence making standard VI methods intractable. We address this gap by introducing variational Gibbs inference (VGI), a new general-purpose method to estimate the parameters of statistical models from incomplete data.')

setup(
    name="cdi",
    author="Vaidotas Simkus",
    description=("Variational Gibbs inference"),
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/vsimkus",
    packages=['cdi']
)
