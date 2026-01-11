FROM python:3.12-slim

WORKDIR /app

# Install requirements
COPY requirements.txt .
RUN python -m pip install --upgrade pip
RUN python -m pip install -r requirements.txt

# Copy all files including trained model
COPY . .

EXPOSE 5001

# Run Flask app
CMD ["python", "app.py"]