FROM python:3.10-slim-buster

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

# Set Streamlit to run properly in Docker
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]