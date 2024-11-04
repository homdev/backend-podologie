# Utiliser une image de base avec Python 3
FROM python:3.10-slim

# Définir le répertoire de travail
WORKDIR /app

# Installation des dépendances système
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    gcc \
    g++ \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Mise à jour de pip et installation des outils de build
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Installation des dépendances scientifiques principales
RUN pip install --no-cache-dir \
    numpy==1.22.4 \
    torch==2.0.1 --extra-index-url https://download.pytorch.org/whl/cpu \
    torchvision==0.15.2 --extra-index-url https://download.pytorch.org/whl/cpu

# Copie et installation des autres dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie du reste des fichiers
COPY . .

# Création des dossiers nécessaires
RUN mkdir -p static/upload static/processed app/models

# Variables d'environnement
ENV PORT=5000
ENV PYTHONUNBUFFERED=1

# Commande de démarrage
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:$PORT", "--workers", "2", "--timeout", "120", "--log-file", "-"]
