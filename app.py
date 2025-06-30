from flask import Flask, request, render_template
import pandas as pd
import joblib
import json

app = Flask(__name__)

# Cargar modelo y columnas
modelo = joblib.load('modelo_random_forest_vinicola_final.pkl')
with open('columnas_modelo.json', 'r') as f:
    columnas = json.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    mes = int(request.form['mes'])
    producto = request.form['producto']
    categoria = request.form['categoria']
    precio = float(request.form['precio'])

    # Crear dataframe con los datos de entrada
    entrada = {
        'Mes': mes,
        'Precio_unitario': precio
    }

    # Variables categóricas one-hot
    for col in columnas:
        if col.startswith('Producto_'):
            entrada[col] = 1 if col == f'Producto_{producto}' else 0
        elif col.startswith('Categoría_'):
            entrada[col] = 1 if col == f'Categoría_{categoria}' else 0

    entrada_df = pd.DataFrame([entrada])
    entrada_df = entrada_df.reindex(columns=columnas, fill_value=0)

    # Realizar la predicción
    prediccion = modelo.predict(entrada_df)[0]

    return render_template('index.html', prediccion=round(prediccion, 2))

if __name__ == '__main__':
    app.run(debug=True)
