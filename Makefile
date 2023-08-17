.PHONY: all

all: dev

dev: install-pkgs install-dev
prod: install-pkgs install-prod

install-pkgs:
	pip install -r requirements.txt

install-dev:
	python setup.py develop

install-prod:
	python setup.py install