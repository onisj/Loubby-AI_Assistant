FROM python:3.10-slim

WORKDIR /app

COPY ./backend/requirements.txt /app/requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY ./backend /app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
