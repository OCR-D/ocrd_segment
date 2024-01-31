FROM ocrd/core:v2.62.0 AS base

WORKDIR /build-ocrd_segment
COPY setup.py .
COPY ocrd_segment/ocrd-tool.json .
COPY ocrd_segment ./ocrd_segment
COPY requirements.txt .
COPY README.md .
RUN pip install . \
	&& rm -rf /build-ocrd_segment

WORKDIR /data
