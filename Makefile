SHELL = /bin/bash
PYTHON ?= python
PIP ?= pip
TAG ?= ocrd/segment

# BEGIN-EVAL makefile-parser --make-help Makefile

help:
	@echo ""
	@echo "  Targets"
	@echo ""
	@echo "    deps     (install required Python packages)"
	@echo "    install  (install this Python package)"
	@echo "    docker   (build Docker image)"
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

docker:
	docker build \
	-t $(TAG) \
	--build-arg VCS_REF=$(git rev-parse --short HEAD) \
	--build-arg BUILD_DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ") .

.PHONY: help deps install docker # deps-test test
