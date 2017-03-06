from setuptools import setup
__version__ = 0.1

desc = ('''Pipeline to turn input text into a w2v embedding.''')

setup(
    name="w2v",
    packages=['word2vec_pipeline'],
    version=__version__,
    author="Travis Hoppe",
    author_email="travis.hoppe+w2v@gmail.com",
    description=desc,
    license = "MIT License",
    keywords = ["NLP", "modeling", "pipeline", ],
    url="https://github.com/NIHOPA/word2vec_pipeline",
    test_suite="tests",
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'miniprez=miniprez.__main__:main',
        ]
    },
    
    # Fill this in when ready...
    download_url='',
)
