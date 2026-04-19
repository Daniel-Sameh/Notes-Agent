FROM python:3.11-slim

# Install basic build tools and dependencies required by ChromaDB and SQLAlchemy
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Run the agent module
CMD ["python", "-m", "src.main"]
