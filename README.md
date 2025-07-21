# 🔢 Digit Recognition using Machine Learning

Este proyecto implementa un sistema de reconocimiento de dígitos escritos a mano usando Python y aprendizaje automático. Entrena un modelo que puede predecir correctamente los dígitos del 0 al 9.

## 🧠 Algoritmo

Se utiliza un modelo de clasificación Random Forest entrenado sobre el dataset de dígitos MNIST 

## 🖼️ Interfaz

La interfaz permite al usuario dibujar un número y recibir una predicción instantánea del modelo.

## ▶️ Cómo ejecutar

1. Instala las dependencias:
   `pip install -r requirements.txt`
   
3. Entrena el modelo (si aún no tienes `digit_model.pkl`):
   `python train_model.py`
   
5. Ejecuta la interfaz:
   `python digit_gui.py`

## 🛠️ Requisitos

- Python 3.10+
- Bibliotecas: `sklearn`, `tkinter`, `joblib`, etc.

## 📌 Notas

- El entorno virtual no se sube al repositorio.
- El modelo entrenado se guarda en `digit_model.pkl`.

## 👨‍💻 Autor

Rodney Piers Salazar Arapa



