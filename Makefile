.PHONY: all

all: install-pkgs install

install-pkgs:
	pip install -r requirements.txt

install:
	python setup.py develop
