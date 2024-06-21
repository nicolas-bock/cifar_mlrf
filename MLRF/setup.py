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
        url='https://github.com/nicolas-bock/cifar_mlrf',
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=[
            'numpy',
            'pandas',
            'seaborn',
            'matplotlib',
            'typer',
            'opencv-python',
            'scikit-learn',
            'loguru',
            'scikit-image',
            'joblib',
            'progressbar2',
            'cv2',
            'python-dotenv',
        ],
        python_requires='>=3.10',
        
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Education",
            "Programming Language :: Python :: 2",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ]
)