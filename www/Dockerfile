FROM python:3
ENV PYTHONUNBUFFERED=1
WORKDIR /clifs
COPY requirements.txt /clifs
RUN pip install -r requirements.txt
COPY . .
WORKDIR /clifs/clifs
