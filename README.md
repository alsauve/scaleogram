# scaleogram

Scaleogram is an easy to go plot tool for data analysis with Continuous Wavelet Transform.

It has the following features:
* simple call signature for complete beginners
* readable axes and clean matplotlib integration
* speed
* portability: tested with python2.7 and python3.7
* comprehensive error message and documentation
* support for Cone Of Influence mask

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

# test with
python -Mscaleogram
```

### Prerequisites

This module depends on
* PyWavelet >= 0.9
* matplotlib >= 2.0
* numpy 1.x


## Running the tests

A features test matrix can be plotted with

```
from scaleogram import test
test()
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



