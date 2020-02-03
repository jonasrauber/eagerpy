.PHONY: test
test:
	pytest --pdb --cov-report term-missing --cov=eagerpy --verbose
	pytest --pdb --cov-report term-missing --cov=eagerpy --cov-append --verbose --backend numpy
	pytest --pdb --cov-report term-missing --cov=eagerpy --cov-append --verbose --backend pytorch
	pytest --pdb --cov-report term-missing --cov=eagerpy --cov-append --verbose --backend jax
	pytest --pdb --cov-report term-missing --cov=eagerpy --cov-append --verbose --backend tensorflow
	pytest --pdb --cov-report term-missing --cov=eagerpy --cov-append --verbose --backend pytorch-gpu

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
	mypy -p eagerpy
	mypy tests/

.PHONY: docs
docs:
	pydocmd generate
	cd docs && vuepress build

.PHONY: servedocs
servedocs:
	cd docs/.vuepress/dist/ && python3 -m http.server 9999

.PHONY: pushdocs
pushdocs:
	cd docs/.vuepress/dist/ && git init && git add -A && git commit -m 'deploy'
	cd docs/.vuepress/dist/ && git push -f git@github.com:jonasrauber/eagerpy.git master:gh-pages

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

.PHONY: pyre
pyre:
	pyre --source-directory . check

.PHONY: pytype
pytype:
	pytype .

.PHONY: pyright
pyright:
	pyright .


.PHONY: mypyreport
mypyreport:
	-mypy . --html-report build
	python3 -m http.server 9999
