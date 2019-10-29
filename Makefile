.PHONY: test
test:
	pytest --verbose

.PHONY: black
black:
	black .

.PHONY: flake8
flake8:
	flake8

.PHONY: mypy
mypy:
	mypy -m eagerpy

.PHONY: install
install:
	pip3 install -e .

.PHONY: devsetup
devsetup:
	pre-commit install

.PHONY: build
build:
	python3 setup.py sdist

.PHONY: release
release: build
	twine upload dist/eagerpy-$(shell cat eagerpy/VERSION).tar.gz
