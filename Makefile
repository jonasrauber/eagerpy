.PHONY: test
test:
	pytest --cov-report term-missing --cov=eagerpy --verbose
	pytest --cov-report term-missing --cov=eagerpy --cov-append --verbose tests/test_main.py --backend numpy
	pytest --cov-report term-missing --cov=eagerpy --cov-append --verbose tests/test_main.py --backend pytorch
	pytest --cov-report term-missing --cov=eagerpy --cov-append --verbose tests/test_main.py --backend jax
	pytest --cov-report term-missing --cov=eagerpy --cov-append --verbose tests/test_main.py --backend tensorflow
	pytest --cov-report term-missing --cov=eagerpy --cov-append --verbose tests/test_main.py --backend pytorch-gpu

.PHONY: black
black:
	black .

.PHONY: blackcheck
blackcheck:
	black --check .

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

.PHONY: commit
commit:
	git add eagerpy/VERSION
	git commit -m 'Version $(shell cat eagerpy/VERSION)'

.PHONY: release
release: build
	twine upload dist/eagerpy-$(shell cat eagerpy/VERSION).tar.gz
