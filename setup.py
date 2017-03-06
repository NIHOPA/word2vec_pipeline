
from setuptools import setup

__version__ = 0.1

setup(
    name="miniprez",
    packages=['miniprez'],
    version=__version__,
    author="Travis Hoppe",
    author_email="travis.hoppe+miniprez@gmail.com",
    description=(
        "Simple markup to web-friendly presentations that look great on mobile and on the big screen."),
    license = "Creative Commons Attribution-ShareAlike 4.0 International License",
    keywords = ["presentations", "reveal.js", "powerpoint", ],
    url="https://github.com/thoppe/miniprez",
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
