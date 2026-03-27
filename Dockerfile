FROM ubuntu:latest

# Avoid interactive prompts during install
ENV DEBIAN_FRONTEND=noninteractive

# Update and install system dependencies
RUN apt update && apt install -y \
    python3 python3-pip python3-venv \
    git curl wget sudo vim \
    libsox-dev libsox-fmt-all freeglut3-dev \
    ca-certificates \
    gnupg \
    apt-transport-https \
    software-properties-common \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install poetry --break-system-packages
WORKDIR /app
COPY pyproject.toml poetry.lock* ./
RUN poetry lock
RUN poetry config virtualenvs.create false
RUN poetry install --no-interaction --no-ansi
COPY . .
CMD ["/bin/bash"]

