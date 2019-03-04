FROM ubuntu:18.04

RUN apt-get update && apt-get install -y python3-pip

WORKDIR /app

COPY requirements.txt /app
COPY synthesized /app/synthesized
COPY web /app/web
RUN touch /app/__init__.py

RUN pip3 install -r requirements.txt
RUN pip3 install gunicorn

ENV SYNTHESIZED_KEY EE6B-6720-67A2-32F3-3139-2DF3-5D2D-B5F3
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

EXPOSE 80

WORKDIR /app

# note that database file is shared and we can run only 1 worker
CMD ["gunicorn", "--bind", "0.0.0.0:80", "--workers", "1", "--threads", "4", "--timeout", "600", "--access-logfile", "-", "web.app:app"]
