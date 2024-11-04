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
    && rm -rf /var/lib/apt/lists/*

# Mise à jour de pip
RUN pip install --no-cache-dir --upgrade pip

# Installation des dépendances principales
RUN pip install --no-cache-dir \
    torch==2.0.1 \
    torchvision==0.15.2 \
    pillow>=9.0.0

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
