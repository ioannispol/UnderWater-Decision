from setuptools import setup, find_packages

setup(
    name='underwater_decision_making',
    version='0.0.1',
    install_requires=[
        'importlib-metadata; python_version >= "3.10"',
        'jupyterlab',
        'setuptools',
        'numpy',
        'scipy',
        'dowhy',
        'matplotlib',
        'pandas',
        'torch ~= 2.0',
        'torchvision',
        'opencv-python <=4.8.0.74',
        'gradio',
        'dowhy',
        'xgboost'
    ],
    extras_require={
        'dev': [
            'pytest',
            'pre-commit',
            'pytest-cov',
            'nbmake'
        ]
    },
    packages=find_packages(
        include=['underwater_decision', 'underwater_decision*'],
        exclude=['tests', 'tests.*', 'notebooks']
    ),
)
