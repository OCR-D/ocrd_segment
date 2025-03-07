SHELL = /bin/bash
PYTHON ?= python
PIP ?= pip
DOCKER_TAG ?= 'ocrd/segment'
DOCKER_BASE_IMAGE ?= docker.io/ocrd/core:v3.1.0

help:
	@echo ""
	@echo "  Targets"
	@echo ""
	@echo "    deps          (install required Python packages)"
	@echo "    install       (install this Python package)"
	@echo "    install-dev   (install in editable mode)"
	@echo "    build         (build source and binary distribution)"
	@echo "    docker        (build Docker image)"
	@echo ""

# END-EVAL

# (install required Python packages)
deps:
	$(PIP) install -r requirements.txt

#deps-test:
#	$(PIP) install -r requirements_test.txt

# Dependencies for deployment in an ubuntu/debian linux
# deps-ubuntu:
# 	sudo apt-get install -y \
# 		...

# (install this Python package)
install: deps
	$(PIP) install .

install-dev: deps
	$(PIP) install -e .

build:
	$(PIP) install build
	$(PYTHON) -m build .

docker:
	docker build \
	-t $(DOCKER_TAG) \
	--build-arg DOCKER_BASE_IMAGE=$(DOCKER_BASE_IMAGE) \
	--build-arg VCS_REF=$(git rev-parse --short HEAD) \
	--build-arg BUILD_DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ") .

.PHONY: help deps install install-dev build docker # deps-test test
