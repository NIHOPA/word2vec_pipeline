# Always prefer setuptools over distutils
import setuptools
import os

__local__ = os.path.abspath(os.path.dirname(__file__))

# Get the long description from the relevant file
long_description = '''word2vec_pipeline
=================================

This is a research and exploration pipeline designed to analyze grants, publication abstracts, and other biomedical corpora. While not designed for production, it is used internally within the Office of Portfolio Analysis at the National Institutes of Health.'''

setuptools.setup(
    name='w2v',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version='0.1.0',

    description='NLP pipeline to parse, embed, and classify with word2vec',
    long_description=long_description,

    # The project's main homepage.
    url="https://github.com/NIHOPA/word2vec_pipeline",

    # Author details
    author="Travis Hoppe",
    author_email="travis.hoppe+w2vec_pipeline@gmail.com",

    # Choose your license
    license='MIT',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2.7',
    ],

    # What does your project relate to?
    keywords="NLP modeling pipeline",

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=['word2vec_pipeline'],

    # Include package data...
    include_package_data=True,

    # entry_points={
    #    'console_scripts': [
    #        'miniprez=miniprez.__main__:main',
    #    ]
    #},

    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    entry_points={
        'console_scripts': [
            'word2vec_pipeline=word2vec_pipeline.__main__:main',
        ],
    },

    test_suite="tests",

    # Fill this in when ready...
    download_url='',
)

'''
setup(
    name="w2v",
    packages=['word2vec_pipeline'],
    version=__version__,

    description=desc,
    license = "MIT License",
    keywords = ["NLP", "modeling", "pipeline", ],
    url="https://github.com/NIHOPA/word2vec_pipeline",
    test_suite="tests",
)
'''
