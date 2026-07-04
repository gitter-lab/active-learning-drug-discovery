from setuptools import setup, find_packages

setup(name='active_learning_dd',
      version='0.1.0',
      description='Cluster-Based Weighted Selector for iterative chemical screening',
      url='https://github.com/gitter-lab/active-learning-drug-discovery',
      author='Moayad Alnammi',
      maintainer='Anthony Gitter',
      maintainer_email='gitter@biostat.wisc.edu',
      license='MIT',
      packages=find_packages(),
      python_requires='==3.6.*',
      zip_safe=False)
