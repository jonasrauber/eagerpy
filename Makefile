test:
	pytest --verbose

black:
	black .

flake8:
	flake8

mypy:
	mypy

install:
	pip3 install -e .
