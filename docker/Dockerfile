# Backend - FastAPI and LangChain
FROM python:3.9

WORKDIR /app

COPY ./requirements.txt /app
RUN pip install -r requirements.txt

COPY ./src /app/src
COPY ./data /app/data
COPY ./frontend/gradio_app.py /app/frontend/gradio_app.py

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
