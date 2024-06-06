# Use the official Python base image
FROM python:3.12.3-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file to the container
COPY ./requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the FastAPI script to the container
COPY app/main.py .

# Start the FastAPI app
CMD ["fastapi", "run", "main.py", "--port", "80"]