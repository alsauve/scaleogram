import setuptools

with open("README.md", "r") as f:
    long_description = f.read()
with open('requirements.txt') as f:
    requirements = f.read().splitlines()
    
setuptools.setup(
     name='scaleogram',  
     version='0.9.3',
     author="Alexandre Sauve",
     author_email="asauve@gmail.com",
     description="User friendly scaleogram plot for Continuous Wavelet Transform",
     long_description=long_description,
     url="https://github.com/alsauve/scaleogram",
     package_dir={'':'lib'},
     packages=['scaleogram'],
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
         "Topic :: Scientific/Engineering :: Information Analysis",
         "Topic :: Scientific/Engineering :: Visualization",
         "Intended Audience :: Information Technology",
         "Intended Audience :: Science/Research",
     ],
     install_requires = requirements,
 )

