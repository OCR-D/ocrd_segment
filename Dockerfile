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
COPY Makefile .
RUN make deps-tf1

COPY setup.py .
COPY ocrd_segment/ocrd-tool.json .
COPY ocrd_segment ./ocrd_segment
COPY maskrcnn_cli ./maskrcnn_cli
COPY requirements.txt .
COPY README.md .
RUN make install
RUN rm -fr /build

WORKDIR /data
VOLUME ["/data"]
