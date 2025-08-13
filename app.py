from flask import Flask, request, render_template
from sentence_transformers import SentenceTransformer, util
import json
import os


app = Flask(__name__)

model = SentenceTransformer('all-mpnet-base-v2')


# Ruta absoluta al JSON (misma carpeta que este script)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(BASE_DIR, "datos.json")

# Cargar el JSON al iniciar la app
with open(JSON_PATH, "r", encoding="utf-8") as f:
    parrafos_cna = json.load(f)  # diccionario en memoria



# Precalcular embeddings para cada párrafo
for p in parrafos_cna:
    p["embedding"] = model.encode(p["texto"], convert_to_tensor=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    resultados = []
    texto_ingresado = ""

    if request.method == 'POST':
        texto_ingresado = request.form.get('actividad', '')
        if texto_ingresado:
            texto_embedding = model.encode(texto_ingresado, convert_to_tensor=True)

            for p in parrafos_cna:
                similitud = util.cos_sim(texto_embedding, p["embedding"]).item()
                resultados.append({
                    "dimension": p["dimension"],
                    "criterio": p["criterio"],
                    "nivel": p["nivel"],
                    "parrafo_num": p["parrafo_num"],
                    "texto": p["texto"],
                    "porcentaje": round(similitud * 100, 2)
                })

            # Ordenar y dejar solo los 3 mejores
            resultados.sort(key=lambda x: x["porcentaje"], reverse=True)
            resultados = resultados[:3]

    return render_template('index.html', resultados=resultados, texto=texto_ingresado)

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5050, debug=True)