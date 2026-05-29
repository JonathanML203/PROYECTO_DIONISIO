from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Cargar el modelo predictor entrenado
try:
    model = joblib.load('modeloPHQ_top11.pkl')
    print("🚀 Microservicio de IA: Modelo .pkl cargado con éxito.")
except Exception as e:
    print(f"⚠️ Alerta: No se pudo cargar el modelo .pkl. Error: {e}")
    model = None

# ENDPOINT ÚNICO: Recibe las respuestas en JSON desde Node.js y devuelve la predicción
@app.route('/api/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({"error": "El modelo predictivo no está disponible en el servidor de IA."}), 500
    
    # Recibir los datos enviados por Node.js
    data = request.get_json()
    if not data or 'respuestas' not in data:
        return jsonify({"error": "Datos inválidos. Se requiere la lista de respuestas."}), 400
    
    vals = data['respuestas'] # Debe ser una lista de 11 números enteros

    if len(vals) != 11:
        return jsonify({"error": f"Se esperaban 11 respuestas, se recibieron {len(vals)}"}), 400

    # Estructurar el DataFrame con los caracteres exactos (\xa0) que exige CatBoost
    df = pd.DataFrame([{
        "Feeling tired or having little energy": vals[0],
        "Moving or speaking so slowly that other people could have noticed\xa0 Or the opposite\x83\xa0 being so fidgety or restless that you have been moving around a lot more than usual": vals[1],
        "Thoughts that you would be better off dead or of hurting yourself in some way": vals[2],
        "Trouble falling or staying asleep, or sleeping too much": vals[3],
        "Trouble concentrating on things, such as reading the newspaper or watching television": vals[4],
        "Little interest or pleasure in doing things": vals[5],
        "Feeling down, depressed, or hopeless": vals[6],
        "Feeling bad about yourself\x83\xa0 or that you are a failure or have let yourself or your family down": vals[7],
        "Financial Pressure": vals[8],
        "Sleep Quality": vals[9],
        "Study Pressure": vals[10]
    }])
    
    # Realizar la predicción con el modelo .pkl
    pred = model.predict(df)
    
    # Regresar el resultado a Node.js en formato de texto para compatibilidad con EJS
    return jsonify({
        "status": "success",
        "prediccion": str(pred[0])
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port)