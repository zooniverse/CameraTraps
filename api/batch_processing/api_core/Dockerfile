FROM python:3.7-slim

ENV LANG=C.UTF-8

WORKDIR /app

RUN apt-get update && apt-get -y upgrade && \
    apt-get install --no-install-recommends -y \
    build-essential && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# install baseline dependencies
RUN pip install poetry

COPY pyproject.toml ./
COPY poetry.toml ./
COPY poetry.lock ./

RUN if echo "development test" | grep -w "$FLASK_ENV"; then \
  poetry install; \
  else poetry install --no-dev; fi

COPY . .

ENV PYTHONUNBUFFERED=1

CMD ["/app/start_flask.sh"]
