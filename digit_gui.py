from tkinter import *
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import joblib 
import os

model = joblib.load("digit_model.pkl")

class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Reconocimiento de Dígitos (Scikit-learn)")

        self.canvas = Canvas(self.root, width=200, height=200, bg='white')
        self.canvas.pack()

        self.btn_predict = Button(self.root, text="Predecir", command=self.predict)
        self.btn_predict.pack()

        self.btn_clear = Button(self.root, text="Limpiar", command=self.clear)
        self.btn_clear.pack()

        self.label_result = Label(self.root, text="", font=("Helvetica", 18))
        self.label_result.pack()

        self.canvas.bind("<B1-Motion>", self.paint)
        self.image = Image.new("L", (200, 200), color=255)
        self.draw = ImageDraw.Draw(self.image)

    def paint(self, event):
        x1, y1 = event.x - 8, event.y - 8
        x2, y2 = event.x + 8, event.y + 8
        self.canvas.create_oval(x1, y1, x2, y2, fill='black')
        self.draw.ellipse([x1, y1, x2, y2], fill=0)

    def predict(self):
        img = self.image.resize((28, 28))
        img = ImageOps.invert(img)
        img = img.point(lambda x: 0 if x < 100 else 255)
        img = np.array(img) / 255.0
        img = img.reshape(1, -1)  # Vector de 784 dimensiones

        pred = model.predict(img)
        self.label_result.config(text=f"Predicción: {pred[0]}")

    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, 200, 200], fill=255)
        self.label_result.config(text="")

if __name__ == "__main__":
    root = Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()
