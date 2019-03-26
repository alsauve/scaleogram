
PYTHON=python3

all: sdist

sdist: clean
	$(PYTHON) setup.py sdist bdist_wheel

upload-test:
	$(PYTHON) -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*

upload-pypi:
	$(PYTHON) -m twine upload dist/*


clean:
	rm -rf ./dist ./lib/scaleogram.egg-info
