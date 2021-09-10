from setuptools import find_packages, setup

# Get the long description from the relevant file
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='dpgen',
    version='0.1',
    description='Automatically generate differentially private programs based on non-private ones.',
    long_description=long_description,
    url='https://github.com/cmla-psu/dpgen',
    author='Yuin Wang,Zeyu Ding,Yingtai Xiao,Daniel Kifer,Danfeng Zhang',
    author_email='zyding@psu.edu,yxwang@psu.edu,yxx5224@psu.edu,dkifer@cse.psu.edu,zhang@cse.psu.edu',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Differential Privacy :: Program Synthesis',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9'
    ],
    keywords='Differential Privacy, Program Synthesis',
    packages=find_packages(exclude=['tests']),
    install_requires=['numpy', 'tqdm', 'numba', 'pyswarms', 'coloredlogs', 'sympy'],
    extras_require={
        'test': ['pytest-cov', 'pytest', 'coverage', 'flaky'],
    },
    entry_points={
        'console_scripts': [
            'dpgen=dpgen.__main__:main',
        ],
    },
)
