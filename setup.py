import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
     name='scaleogram',  
     version='0.9',
     author="Alexandre Sauve",
     author_email="asauve@gmail.com",
     description="Easy scaleogram plot for Continuous Wavelet Transform",
     long_description=long_description,
     url="https://github.com/alsauve/scaleogram",
     package_dir={'':'lib'},
     packages=['scaleogram'],
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )

