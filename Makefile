.PHONY: all

all: dev

dev: install-pkgs install-dev
prod: install-pkgs install-prod

install-pkgs:
	pip install --extra-index-url http://24.199.104.228/simple --trusted-host 24.199.104.228 torchsparse==2.1.0+torch110cu113 --force-reinstall
	pip install -r requirements.txt

install-dev:
	python setup.py develop

install-prod:
	python setup.py install