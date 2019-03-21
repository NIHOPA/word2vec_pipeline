# Always prefer setuptools over distutils
import setuptools
import os

__local__ = os.path.abspath(os.path.dirname(__file__))
f_version = os.path.join(__local__, 'embedding_pipeline', '_version.py')
exec(open(f_version).read())


# Get the long description from the relevant file

long_description = '''word2vec_pipeline
=================================

This is a research and exploration pipeline designed to analyze grants, publication abstracts, and other biomedical corpora. While not designed for production, it is used internally within the Office of Portfolio Analysis at the National Institutes of Health.'''  # NOQA

setuptools.setup(
    name='word2vec_pipeline',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version=__version__,

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
        #'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.7',
    ],

    # What does your project relate to?
    keywords="NLP modeling pipeline",

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=['word2vec_pipeline'],

    # Include package data...
    include_package_data=True,

    entry_points={
        'console_scripts': [
            'word2vec_pipeline=word2vec_pipeline.__main__:main',
        ],
    },

    test_suite="tests",

    # Fill this in when ready...
    download_url='',
)
