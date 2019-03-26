# scaleogram

Scaleogram is a user friendly plot tool for 1D data analysis with Continuous Wavelet Transform. 

I started this project when realizing how harsh it can be to build nice plots
of wavelets scaleogram with axes ticks and labels consistent with the actual location of features.
Hence this module aim to provide a reliable tool for either quick data analysis or publication.

It has the following features:

* simple call signature for complete beginners

* readable axes and clean matplotlib integration

* many options for changing scale, spectrum filter, colorbar integration, etc...

* support for periodicity and frequency units, consistent with labelling

* speed

* portability: tested with python2.7 and python3.7

* comprehensive error messages and documentation with examples

* support for [Cone Of Influence]() mask


## Install with pip

Installation should be straightforward with

```
pip install scaleogram
```

## Install from github

```
git clone http://github.com/alsauve/scaleogram
cd scaleogram
python ./setup.py install --user
```

### Prerequisites

This module depends on

* PyWavelet >= 0.9
* matplotlib >= 2.0
* numpy >= 1.0

## Documentation

A lot of documentation and examples are available online from the docstrings

Jupyter notebook are also provided for quickstarting

* A gentle introduction to CWT based data analysis [TODO]
* [scale to frequency relationship](https://github.com/alsauve/scaleogram/blob/master/doc/scale-to-frequency.ipynb)
* [Example of scaleogram with the NINO3.4 SST seasonal time series](https://github.com/alsauve/scaleogram/blob/master/doc/El-Nino-Dataset.ipynb)
* [Graphical output of the test set](https://github.com/alsauve/scaleogram/blob/master/doc/tests.ipynb)


## Running the tests

A features test matrix can be plotted with

```
# launch graphical tests
python -m scaleogram.test
```

## Built With

* [ViM](https://www.vim.org/) - The editor
* [Spyder](https://www.spyder-ide.org/) - The Scientific Python Developement Environment
* [Jupyter](https://jupyter.org/) - The Jupyter Notebook

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/alsauve/scaleogram/tags). 

## Authors

* **Alexandre sauve** - *Initial work* - [Scaleogram](https://github.com/alsauve/scaleogram)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* The team behind PyWavelet for their nice job into making wavelet transform available
* The Matlab environement for inspiration and good documentation
* Mabel Calim Costa for the waipy package and inspiration



