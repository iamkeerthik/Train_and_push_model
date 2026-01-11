FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN python -m pip install --upgrade pip
RUN python -m pip install -r requirements.txt

# Copy model code + artifacts (trained model)
COPY . .

EXPOSE 5001

CMD ["python", "app.py"]