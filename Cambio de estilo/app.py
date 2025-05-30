from flask import Flask, request, send_file
from flask_cors import CORS
from io import BytesIO
import torch
import os

# Importa tus funciones locales
from infer import load_generator, run_inference  # Asegúrate de tener infer.py en la misma carpeta

app = Flask(__name__)
CORS(app)
device = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------
# Cargar modelos una sola vez al iniciar
# --------------------------
# G_BARROCO: convierte cualquier fachada al estilo Barroco
# G_GOTICO: convierte cualquier fachada al estilo Gótico
G_GOTICO = load_generator("results/G_AB_100.pth", device)     # A → Gótico
G_BARROCO = load_generator("results/G_BA_100.pth", device)    # A → Barroco

# --------------------------
# Ruta para recibir y procesar imagen
# --------------------------
@app.route("/generate", methods=["POST"])
def generate():
    if "image" not in request.files:
        return "Imagen no proporcionada", 400

    image_file = request.files["image"]
    target_style = request.form.get("style", "barroco").lower()  # 'barroco' o 'gotico'

    # Verifica que el estilo sea válido
    if target_style not in ["barroco", "gotico"]:
        return "Estilo solicitado no válido. Usa 'barroco' o 'gotico'.", 400

    # Guardar imagen temporalmente
    input_path = "temp_input.jpg"
    output_path = "temp_output.jpg"
    image_file.save(input_path)

    # Elegir modelo según estilo destino
    model = G_BARROCO if target_style == "barroco" else G_GOTICO

    # Ejecutar inferencia y guardar resultado
    run_inference(input_path, model, output_path, device)

    # Enviar imagen generada como respuesta
    return send_file(output_path, mimetype="image/jpeg")

# --------------------------
# Ejecutar servidor
# --------------------------
if __name__ == "__main__":
    app.run(debug=True)