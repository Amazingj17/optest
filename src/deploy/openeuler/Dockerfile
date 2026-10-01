ARG BASE_IMAGE=openeuler/openeuler:24.03-lts-sp4@sha256:17c15554be2a5bc46023acb6e04d609d77642b8c20e236e88deb18e41ae4558e
FROM ${BASE_IMAGE}

RUN dnf install -y python3-pip git libgomp && dnf clean all
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv ${VIRTUAL_ENV}
ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    MPLBACKEND=Agg \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1

COPY deploy/openeuler/requirements.txt /tmp/openeuler-requirements.txt
RUN python -m pip install --no-cache-dir pip==25.3 && \
    python -m pip install --no-cache-dir torch==2.9.1+cpu --index-url https://download.pytorch.org/whl/cpu && \
    python -m pip install --no-cache-dir -r /tmp/openeuler-requirements.txt

WORKDIR /opt/optest
COPY . .
RUN python -m pip install --no-deps --no-build-isolation -e .
CMD ["python", "scripts/verify_platform.py", "--output", "outputs/openeuler_acceptance", "--execution-kind", "container"]
