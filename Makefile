
PYTHON=python3

all: sdist

sdist: clean
	$(PYTHON) setup.py sdist bdist_wheel

upload-test:
	$(PYTHON) -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*

upload-pypi:
	$(PYTHON) -m twine upload dist/*

pip-install-test:
	PIP_IGNORE_INSTALLED=0 $(PYTHON) -m pip install --user --index-url https://test.pypi.org/simple/  scaleogram

pip-install:
	PIP_IGNORE_INSTALLED=0 $(PYTHON) -m pip install --user scaleogram

pip-uninstall:
	$(PYTHON) -m pip uninstall scaleogram


clean:
	rm -rf ./dist ./lib/scaleogram.egg-info
