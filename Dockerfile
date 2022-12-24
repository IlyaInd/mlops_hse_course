FROM python:3.10
COPY core ./app/core
COPY main.py poetry.lock pyproject.toml ./app/
WORKDIR /app
RUN pip install poetry==1.2
RUN poetry install --without dev
CMD ["poetry", "run", "python", "main.py"]