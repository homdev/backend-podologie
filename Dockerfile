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

# Copie des fichiers de requirements
COPY requirements.txt .

# Installation des dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copie du reste des fichiers
COPY . .

# Création des dossiers nécessaires
RUN mkdir -p static/upload static/processed app/models

# Exposer le port
EXPOSE $PORT

# Commande de démarrage
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:$PORT", "--workers", "2", "--timeout", "120", "--log-file", "-"]
