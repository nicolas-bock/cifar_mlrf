from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'CIFAR-10 Python package'
LONG_DESCRIPTION = 'Python package for CIFAR-10 dataset and model training.'

# Setting up
setup(
        name="cifar_package", 
        version=VERSION,
        author="Nicolas BOCK",
        author_email="<nicolas.bock@epita.fr>",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=[],
        
        keywords=['python', 'cifar'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Education",
            "Programming Language :: Python :: 2",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ]
)