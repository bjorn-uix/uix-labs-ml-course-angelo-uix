FROM python:3.9
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
ENV FLASK_RUN_HOST=0.0.0.0
CMD ["flask", "run", "--host", "0.0.0.0", "--port", "80"]
