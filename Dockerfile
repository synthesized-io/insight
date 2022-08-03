FROM python:3.8 as base
RUN apt-get update
RUN pip install -U pip setuptools wheel

FROM base as builder
WORKDIR /build
RUN pip install build setuptools_scm
COPY pyproject.toml README.md /build/
COPY .git /build/.git
COPY src /build/src
RUN python -m build -nw --outdir /dist

FROM base
WORKDIR /code
RUN apt-get install libpq-dev postgresql-client -y
COPY requirements.txt requirements.txt
RUN pip install -U pip wheel
RUN pip install -r requirements.txt
COPY alembic.ini /code/
COPY migrations /code/migrations
COPY scripts /code/scripts
COPY --from=builder /dist/*.whl /dist/
RUN pip install /dist/*.whl
CMD /bin/bash
