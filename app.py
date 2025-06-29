from flask import Flask, request, render_template
import pandas as pd
import joblib
import json

app = Flask(__name__)

# Cargar modelo
modelo = joblib.load('modelo_random_forest_vinicola_final.pkl')

# Cargar columnas desde el JSON
with open('columnas_modelo.json', 'r') as f:
    columnas_entrenamiento = json.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        precio = float(request.form['precio'])
        descuento = float(request.form['descuento'])
        stock = int(request.form['stock'])
        mes = int(request.form['mes'])

        metodo_pago = request.form['metodo_pago']
        region = request.form['region']
        dia_semana = request.form['dia_semana']
        categoria = request.form['categoria']
        evento = request.form['evento']
        producto = request.form['producto']

        entrada = {
            'Precio_unitario': precio,
            'Descuento_%': descuento,
            'Stock_disponible': stock,
            'Mes': mes,
            # Método de pago
            'Método_pago_Tarjeta': 1 if metodo_pago == 'Tarjeta' else 0,
            'Método_pago_Transferencia': 1 if metodo_pago == 'Transferencia' else 0,
            'Método_pago_Yape/Plin': 1 if metodo_pago == 'Yape/Plin' else 0,
            # Región
            'Región_Trujillo': 1 if region == 'Trujillo' else 0,
            'Región_Lima': 1 if region == 'Lima' else 0,
            'Región_Cusco': 1 if region == 'Cusco' else 0,
            'Región_Piura': 1 if region == 'Piura' else 0,
            # Día de la semana
            'Día_de_semana_Monday': 1 if dia_semana == 'Monday' else 0,
            'Día_de_semana_Tuesday': 1 if dia_semana == 'Tuesday' else 0,
            'Día_de_semana_Wednesday': 1 if dia_semana == 'Wednesday' else 0,
            'Día_de_semana_Thursday': 1 if dia_semana == 'Thursday' else 0,
            'Día_de_semana_Saturday': 1 if dia_semana == 'Saturday' else 0,
            'Día_de_semana_Sunday': 1 if dia_semana == 'Sunday' else 0,
            # Categoría
            'Categoría_Dulce': 1 if categoria == 'Dulce' else 0,
            'Categoría_Espumante': 1 if categoria == 'Espumante' else 0,
            'Categoría_Rosé': 1 if categoria == 'Rosé' else 0,
            'Categoría_Tinto': 1 if categoria == 'Tinto' else 0,
            # Evento especial
            'Evento_especial_Día del Padre': 1 if evento == 'Día del Padre' else 0,
            'Evento_especial_Feria del Vino': 1 if evento == 'Feria del Vino' else 0,
            'Evento_especial_Fiestas Patrias': 1 if evento == 'Fiestas Patrias' else 0,
            'Evento_especial_Vendimia': 1 if evento == 'Vendimia' else 0,
            # Producto
            'Producto_Mistela': 1 if producto == 'Mistela' else 0,
            'Producto_Vino Blanco Dulce': 1 if producto == 'Vino Blanco Dulce' else 0,
            'Producto_Vino Blanco Seco': 1 if producto == 'Vino Blanco Seco' else 0,
            'Producto_Vino Reservado': 1 if producto == 'Vino Reservado' else 0,
            'Producto_Vino Rosé': 1 if producto == 'Vino Rosé' else 0,
            'Producto_Vino Tinto Dulce': 1 if producto == 'Vino Tinto Dulce' else 0,
            'Producto_Vino Tinto Seco': 1 if producto == 'Vino Tinto Seco' else 0,
            'Producto_Vino Tinto Semi-seco': 1 if producto == 'Vino Tinto Semi-seco' else 0
        }

        entrada_df = pd.DataFrame([entrada])

        # Asegurar que las columnas estén en el mismo orden que en el entrenamiento
        entrada_df = entrada_df[columnas_entrenamiento]

        prediccion = modelo.predict(entrada_df)[0]

        return render_template('index.html', prediccion=round(prediccion, 2))

if __name__ == '__main__':
    app.run(debug=True)
