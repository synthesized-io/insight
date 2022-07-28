FROM python:3.7-alpine as base
RUN apk add build-base
RUN pip install -U pip setuptools wheel

FROM base as builder
WORKDIR /build
RUN apk add git
RUN pip install build setuptools_scm
COPY pyproject.toml README.md /build/
COPY .git /build/.git
COPY src /build/src
RUN python -m build -nw --outdir /dist

FROM base
WORKDIR /code
RUN apk add postgresql-dev postgresql-client bash
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY alembic.ini /code/
COPY migrations /code/migrations
COPY tests /code/tests
COPY --from=builder /dist/*.whl /dist/
RUN pip install /dist/*.whl
CMD /bin/bash
