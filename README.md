# 🏛️ Architectural Style Transfer – Barroco y Gótico

Este proyecto consiste en una aplicación web que permite **transformar imágenes de fachadas arquitectónicas** comunes hacia **estilos clásicos** como el **barroco** o el **gótico**, utilizando redes generativas adversariales (**CycleGAN**).

🎯 **Enfoque educativo**: El objetivo no es comercial, sino **enseñar visualmente las diferencias estilísticas** entre periodos arquitectónicos. En versiones futuras, se mostrará una **descripción del estilo** junto a la imagen generada.

---

## 📚 Descripción del proyecto

Esta herramienta recibe una imagen de fachada y, mediante un modelo entrenado con **CycleGAN**, la transforma visualmente hacia el estilo seleccionado por el usuario. El modelo ha sido entrenado con imágenes reales de estilos arquitectónicos extraídas del dataset de Kaggle.

El proyecto incluye:

- 🧠 Un modelo **CycleGAN entrenado en PyTorch**
- 🧪 Un backend en **Flask** para manejar la inferencia
- 🌐 Un frontend en **React** para la interfaz de usuario

---

## 🧱 Estructura del proyecto

    proyecto-pdi/
    ├── app.py # Servidor Flask (API)
    ├── infer.py # Módulo de inferencia con el modelo entrenado
    ├── train_cyclegan.py # Entrenamiento de los modelos
    ├── results/ # Modelos entrenados (.pth)
    │ ├── G_AB_100.pth
    │ └── G_BA_100.pth
    ├── data/ # Dataset preparado
    │ ├── A/ (Barroco)
    │ └── B/ (Gótico)
    ├── frontend/ # Carpeta del proyecto React
    │ ├── App.js
    │ └── App.css
    ├── requirements.txt
    └── README.md

---

## ⚙️ Tecnologías utilizadas

- Python 3.10+
- PyTorch
- Flask + Flask-CORS
- React (Vite o Create React App)
- HTML + CSS (minimalista y responsivo)

---

## 🚀 ¿Cómo ejecutar el proyecto localmente?

### 1. Clona el repositorio

git clone https://github.com/tuusuario/proyecto-pdi.git
cd proyecto-pdi

### 2. Instala dependencias del backend

Dentro del proyecto existe un archivo con los requerimientos y puedes instalarlo así:
pip install -r requirements.txt

O puedes usar el comando, para instalar todas las librerías por ti mismo:
pip install flask flask-cors torch torchvision pillow tqdm

### 3. Ejecuta el servidor

python app.py

El servidor estará disponible en http://localhost:5000

### 4. Ejecuta el Frontend

cd frontend
npm install
npm start

La aplicación estará disponible en http://localhost:3000

---

# 🧠 Modelo y entrenamiento

Arquitectura: CycleGAN con 2 generadores (G_AB_, G_BA_) y 2 discriminadores (D_A_, D_B_)

Resolución: 256×256

Épocas: 100

Tiempo de entrenamiento: ~7 horas

Batch size: 1 (ajustado por limitaciones de VRAM)

Dataset usado: Architectural Styles Dataset
(Kaggle: https://www.kaggle.com/datasets/dumitrux/architectural-styles-dataset)

Se filtraron solo dos categorías para el PMV.

Disparidad en el dataset: 456 imágenes barrocas, 331 góticas.

En esta versión se entrenó una GAN por dirección de estilo, y solo se realizó una corrida de entrenamiento completa por limitaciones de hardware.

---

## 💡 Características futuras (por implementar)

- Descripción educativa automática del estilo seleccionado

- Soporte para más estilos arquitectónicos (neoclásico, art decó, etc.)

- Segmentación de fachada para mayor precisión

- Mejoras de resolución

---

# 📜 Créditos

Desarrollado como proyecto final para la asignatura Procesamiento Digital de Imágenes (PDI), Ingeniería Multimedia – Universidad Autónoma de Occidente.

Por: Sebastián Giraldo García
2025

---

# 📎 Licencia

Uso educativo, sin fines comerciales. Dataset con licencia de uso público en Kaggle.
