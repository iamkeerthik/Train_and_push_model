FROM python:3.12-slim

WORKDIR /app

# Install requirements
COPY requirements.txt .
RUN python -m pip install --upgrade pip
RUN python -m pip install -r requirements.txt

# Copy model folder including trained artifacts
COPY . .

EXPOSE 5001

CMD ["python", "app.py"]