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

# Workaround for missing prebuilt versions of TF<2 for Python==3.8
# todo: find another solution for 3.9, 3.10 etc
# https://docs.nvidia.com/deeplearning/frameworks/tensorflow-wheel-release-notes/tf-wheel-rel.html
# Nvidia has them, but under a different name, so let's rewrite that:
# (hold at nv22.11, because newer releases require CUDA 12, which is not supported by TF2 (at py38),
#  and therefore not in our ocrd/core-cuda base image yet)
# However, at that time no Numpy 1.24 was known, which breaks TF1
# (which is why later nv versions hold it at <1.24 automatically -
#  see https://github.com/NVIDIA/tensorflow/blob/r1.15.5%2Bnv22.11/tensorflow/tools/pip_package/setup.py)
deps-tf1:
	if $(PYTHON) -c 'import sys; print("%u.%u" % (sys.version_info.major, sys.version_info.minor))' | fgrep 3.8 && \
	! $(PIP) show -q tensorflow-gpu; then \
	  $(PIP) install nvidia-pyindex && \
	  pushd $$(mktemp -d) && \
	  $(PIP) download --no-deps nvidia-tensorflow==1.15.5+nv22.11 && \
	  for name in nvidia_tensorflow-*.whl; do name=$${name%.whl}; done && \
	  $(PYTHON) -m wheel unpack $$name.whl && \
	  for name in nvidia_tensorflow-*/; do name=$${name%/}; done && \
	  newname=$${name/nvidia_tensorflow/tensorflow_gpu} &&\
	  sed -i s/nvidia_tensorflow/tensorflow_gpu/g $$name/$$name.dist-info/METADATA && \
	  sed -i s/nvidia_tensorflow/tensorflow_gpu/g $$name/$$name.dist-info/RECORD && \
	  sed -i s/nvidia_tensorflow/tensorflow_gpu/g $$name/tensorflow_core/tools/pip_package/setup.py && \
	  pushd $$name && for path in $$name*; do mv $$path $${path/$$name/$$newname}; done && popd && \
	  $(PYTHON) -m wheel pack $$name && \
	  $(PIP) install $$newname*.whl && popd && rm -fr $$OLDPWD; \
	  $(PIP) install "numpy<1.24"; \
	  $(PIP) install imageio==2.14.1 "tifffile<2022"; \
	  $(PIP) install --no-binary imgaug imgaug; \
fi

deps: deps-tf1
	$(PIP) install -r requirements.txt

install: deps
	$(PIP) install .

docker:
	docker build \
	-t $(TAG) \
	--build-arg VCS_REF=$(git rev-parse --short HEAD) \
	--build-arg BUILD_DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ") .

.PHONY: help deps deps-tf1 install docker # deps-test test
