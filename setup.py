#!/usr/bin/env python

import os
from setuptools import setup

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

version_fname = os.path.join(THIS_DIR, 'flowws_structure_pretraining', 'version.py')
with open(version_fname) as version_file:
    exec(version_file.read())

readme_fname = os.path.join(THIS_DIR, 'README.md')
with open(readme_fname) as readme_file:
    long_description = readme_file.read()

entry_points = set()
flowws_modules = []
package_names = ['flowws_structure_pretraining']

def add_subpkg(subpkg, module_names):
    package_names.append('flowws_structure_pretraining.{}'.format(subpkg))
    for name in module_names:
        if name not in entry_points:
            flowws_modules.append('{0} = flowws_structure_pretraining.{1}.{0}:{0}'.format(name, subpkg))
            entry_points.add(name)
        flowws_modules.append(
            'flowws_structure_pretraining.{1}.{0} = flowws_structure_pretraining.{1}.{0}:{0}'.format(name, subpkg))

module_names = [
    'ClearMetrics',
    'DistanceNeighbors',
    'FileLoader',
    'LimitAccuracyCallback',
    'LoadModel',
    'NearestNeighbors',
    'NormalizeNeighborDistance',
    'PyriodicLoader',
    'SANNeighbors',
    'VoronoiNeighbors',
]
for name in module_names:
    if name not in entry_points:
        flowws_modules.append('{0} = flowws_structure_pretraining.{0}:{0}'.format(name))
        entry_points.add(name)
    flowws_modules.append(
        'flowws_structure_pretraining.{0} = flowws_structure_pretraining.{0}:{0}'.format(name))

subpkg = 'analysis'
module_names = [
    'AutoencoderVisualizer',
    'BondDenoisingVisualizer',
    'ClassifierPlotter',
    'EmbeddingDistance',
    'EmbeddingDistanceTrajectory',
    'EmbeddingPlotter',
    'EvaluateEmbedding',
    'NoisyBondVisualizer',
    'PCAEmbedding',
    'RegressorPlotter',
    'ShiftIdentificationVisualizer',
    'UMAPEmbedding',
]
add_subpkg(subpkg, module_names)

subpkg = 'models'
module_names = [
    'GalaBondClassifier',
    'GalaBondRegressor',
    'GalaBottleneckAutoencoder',
    'GalaScalarRegressor',
    'GalaVectorAutoencoder',
]
add_subpkg(subpkg, module_names)

subpkg = 'tasks'
module_names = [
    'AutoencoderTask',
    'DenoisingTask',
    'FrameClassificationTask',
    'FrameRegressionTask',
    'NearestBondTask',
    'NoisyBondTask',
    'ShiftIdentificationTask',
]
add_subpkg(subpkg, module_names)

setup(name='flowws-structure-pretraining',
      author='Matthew Spellings',
      author_email='mspells@vectorinstitute.ai',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3',
      ],
      description='Stage-based scientific workflows for pretraining deep learning models on self-assembly data',
      entry_points={
          'flowws_modules': flowws_modules,
      },
      extras_require={
          'pyriodic': ['pyriodic-aflow'],
          'analysis': [
              'plato-draw',
              'pynndescent',
              'scikit-learn',
              'umap-learn',
          ],
      },
      install_requires=[
          'flowws',
          'flowws-keras-experimental',
          'freud-analysis',
          'garnett',
          'geometric-algebra-attention',
          'gtar',
          'pyriodic-structures',
          'rowan',
      ],
      license='MIT',
      long_description=long_description,
      long_description_content_type='text/markdown',
      packages=package_names,
      python_requires='>=3',
      version=__version__
      )
