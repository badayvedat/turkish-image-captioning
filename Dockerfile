# syntax=docker/dockerfile:1

FROM python:3.7

WORKDIR /app

COPY requirements.txt requirements.txt
COPY . .
RUN pip3 install -r requirements.txt

WORKDIR /app/app
CMD ["flask", "run", "--host=0.0.0.0"]
