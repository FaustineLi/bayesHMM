from setuptools import setup

setup(
    name='bayesHMM',
    version='0.1',
    description='Bayesian Hidden Markov Models in Python',
    author='Faustine Li',
    license='MIT',
    packages=['bayesHMM'],
    install_requires=[
        'numpy>=1.12',
        'scipy>=0.9',
        'tqdm'
    ]
)