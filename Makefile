.PHONY: test
test:
	pytest --verbose --ignore=eagerpy/tests/test_jax_tensor.py
	pytest --verbose eagerpy/tests/test_jax_tensor.py

.PHONY: testcov
testcov:
	pytest --cov-report term-missing --cov=eagerpy --verbose --ignore=eagerpy/tests/test_jax_tensor.py
	pytest --cov-report term-missing --cov=eagerpy --cov-append --verbose eagerpy/tests/test_jax_tensor.py

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

.PHONY: commit
commit:
	git add eagerpy/VERSION
	git commit -m 'Version $(shell cat eagerpy/VERSION)'

.PHONY: release
release: build
	twine upload dist/eagerpy-$(shell cat eagerpy/VERSION).tar.gz
