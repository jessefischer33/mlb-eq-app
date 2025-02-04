# Use Python base image
FROM python:3.10

# Set the working directory
WORKDIR /app

# Copy the application code
COPY . .

# Install dependencies
RUN pip install -r requirements.txt

# Expose port
EXPOSE 8080

# Run the app
CMD ["python", "app.py"]
