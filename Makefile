SHELL = /bin/bash
PYTHON ?= python
PIP ?= pip
TAG ?= ocrd/segment:maskrcnn-cli

help:
	@echo ""
	@echo "  Targets"
	@echo ""
	@echo "    deps     (install required Python packages)"
	@echo "    install  (install this Python package)"
	@echo "    docker   (build Docker image)"
	@echo ""

deps:
	$(PIP) install -r requirements.txt

install: deps
	$(PIP) install .

docker:
	docker build \
	-t $(TAG) \
	--build-arg VCS_REF=$(git rev-parse --short HEAD) \
	--build-arg BUILD_DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ") .

.PHONY: help deps install docker # deps-test test
