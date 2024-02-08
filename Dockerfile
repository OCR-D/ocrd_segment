FROM ocrd/core:v2.62.0 AS base
ARG VCS_REF
ARG BUILD_DATE
LABEL \
    maintainer="https://github.com/OCR-D/ocrd_segment/issues" \
    org.label-schema.vcs-ref=$VCS_REF \
    org.label-schema.vcs-url="https://github.com/OCR-D/ocrd_segment" \
    org.label-schema.build-date=$BUILD_DATE

WORKDIR /build
COPY setup.py .
COPY ocrd_segment/ocrd-tool.json .
COPY ocrd_segment ./ocrd_segment
COPY requirements.txt .
COPY README.md .
RUN pip install .
RUN rm -rf /build

WORKDIR /data
VOLUME ["/data"]
