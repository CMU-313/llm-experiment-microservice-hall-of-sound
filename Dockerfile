FROM python:3.12

# Set working directory
WORKDIR /app

ENV FLASK_APP=app.py

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Expose the port Flask runs on
EXPOSE 5000

# Long-running HTTP API for downstream callers
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]