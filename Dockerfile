FROM python:3.10
#COPY core ./app/core
#COPY db.py main.py poetry.lock pyproject.toml ./app/
VOLUME . /app
WORKDIR /app
#RUN pip install poetry==1.1.15
#RUN poetry install
#CMD poetry run python db.py main.py