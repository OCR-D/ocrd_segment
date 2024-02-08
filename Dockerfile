FROM ocrd/core-cuda:v2.62.0 AS base
ARG VCS_REF
ARG BUILD_DATE
LABEL \
    maintainer="https://github.com/OCR-D/ocrd_segment/issues" \
    org.label-schema.vcs-ref=$VCS_REF \
    org.label-schema.vcs-url="https://github.com/OCR-D/ocrd_segment" \
    org.label-schema.build-date=$BUILD_DATE

SHELL ["/bin/bash", "-c"]
WORKDIR /build
RUN pip install nvidia-pyindex && \
    pushd $(mktemp -d) && \
    pip download --no-deps "nvidia-tensorflow==1.15.5+nv22.12" && \
    for name in nvidia_tensorflow-*.whl; do name=${name%.whl}; done && \
        python -m wheel unpack $name.whl && \
        for name in nvidia_tensorflow-*/; do name=${name%/}; done && \
        newname=${name/nvidia_tensorflow/tensorflow_gpu} &&\
        sed -i s/nvidia_tensorflow/tensorflow_gpu/g $name/$name.dist-info/METADATA && \
        sed -i s/nvidia_tensorflow/tensorflow_gpu/g $name/$name.dist-info/RECORD && \
        sed -i s/nvidia_tensorflow/tensorflow_gpu/g $name/tensorflow_core/tools/pip_package/setup.py && \
        pushd $name && for path in $name*; do mv $path ${path/$name/$newname}; done && popd && \
        python -m wheel pack $name && \
        pip install $newname*.whl && popd && rm -fr $OLDPWD

# - preempt conflict over numpy between scikit-image and tensorflow
# - preempt conflict over numpy between tifffile and tensorflow (and allow py36)
RUN pip install imageio==2.14.1 "tifffile<2022"
# - preempt conflict over numpy between h5py and tensorflow
RUN pip install "numpy<1.24"
# - preempt imgaug wheel from dragging in opencv-python instead of opencv-python-headless
RUN pip install --no-binary imgaug imgaug

COPY setup.py .
COPY ocrd_segment/ocrd-tool.json .
COPY ocrd_segment ./ocrd_segment
COPY maskrcnn_cli ./maskrcnn_cli
COPY requirements.txt .
COPY README.md .
RUN pip install .
RUN rm -fr /build

WORKDIR /data
VOLUME ["/data"]
