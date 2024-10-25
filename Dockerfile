# Utiliser une image de base avec Python 3
FROM python:3.10-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers de l'application dans le conteneur
COPY . /app

# Installer les dépendances système requises
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Exposer le port sur lequel Flask va tourner
EXPOSE 5000

# Lancer l'application Flask
CMD ["python3", "app.py"]
