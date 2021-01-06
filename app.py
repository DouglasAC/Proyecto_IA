from flask import Flask, render_template, request, json, jsonify
import pickle
from datetime import datetime
import numpy as np
from Neural_Network.Data import Data

app = Flask(__name__)

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0



@app.route('/')
def inicio():
    return render_template('index.html')


@app.route('/datos', methods=['GET', 'POST', 'DELETE', 'PUT'])
def datos():
    departamentos = cargar("Guardar/Datos.pkl")
    print("Edad:", departamentos.maxEdad,  departamentos.minEdad, "Año:", departamentos.maxYear, departamentos.minYear, "Distancia: ", departamentos.maxDist, departamentos.minDist)
    valores = []
    for dep in departamentos.dep_list:
        nom = dep.nombre
        municipio = []
        for muni in dep.municipios:
            municipio.append([muni.nombre, muni.distancia])
        valores.append([nom, municipio])
    return jsonify(
        depa = valores
    )


@app.route('/predecir', methods=['GET', 'POST', 'DELETE', 'PUT'])
def predecir():
    data = request.get_json()
    genero = int(data["genero"])
    edad = float(data["edad"])
    year = float(data["year"])
    distancia = float(data["distancia"])

    print(genero, edad, year, distancia)
    mejor = cargar("Guardar/Mejor.pkl")
    departamentos = cargar("Guardar/Datos.pkl")
    esc_edad = (edad - departamentos.minEdad) / (departamentos.maxEdad - departamentos.minEdad)
    esc_year = (year - departamentos.minYear) / (departamentos.maxYear - departamentos.minYear)
    esc_dist = (distancia - departamentos.minDist) / (departamentos.maxDist - departamentos.minDist)
    print(genero, esc_edad, esc_year, esc_dist)
    # [Genereo, Edad, Año, Distancia]
    valor = [genero, esc_edad, esc_year, esc_dist]
    datos_arr = np.asarray([valor])
    resp_arr = np.asarray([[0]])
    datos_entrenamiento = datos_arr.T
    respuesta_entrenamiento = resp_arr.T
    print(datos_arr.shape)
    print(datos_entrenamiento.shape)
    print(resp_arr.shape)
    print(respuesta_entrenamiento.shape)
    test_set = Data(datos_entrenamiento, respuesta_entrenamiento)
    val = predecirModelo(mejor.modelo, test_set)
    resp = "Activo" if val == 1 else "Traslado"
    return jsonify(
        depa = resp
    )


def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def cargar(filename):
    with open(filename, 'rb') as input:
        obj = pickle.load(input)
    return obj

def predecirModelo(modelo, dataSet):
    # Se obtienen los datos
    m = dataSet.m
    Y = dataSet.y
    p = np.zeros((1, m), dtype= np.int)
    # Propagacion hacia adelante
    y_hat, temp = modelo.propagacion_adelante(dataSet)
    # Convertir probabilidad
    for i in range(0, m):
        print("y_hat:",y_hat )
        p[0, i] = 1 if y_hat[0, i] > 0.5 else 0
    print(p)
    exactitud = np.mean((p[0, :] == Y[0, ]))
    print("Exactitud: " + str(exactitud))
    return p[0][0]