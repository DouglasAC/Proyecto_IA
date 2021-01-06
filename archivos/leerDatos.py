import csv
from archivos.Municipio import Municipio
from archivos.Departamento import Departamento
import math
import numpy as np
import random

def leerMunicipios():
    with open('archivos/Municipios.csv', newline='', encoding='utf-8') as csvfile:
        spamreader = csv.reader(csvfile)
        valores = list(spamreader)
        del valores[0]
        deplista = []
        for valor in valores:
            if not existeDepartamento(valor[0], deplista):
                dep = Departamento(valor[0])
                distancia = haversine(
                    14.589246, -90.551449, float(valor[3]), float(valor[4]))
                mun = Municipio(valor[2], valor[1], float(
                    valor[3]), float(valor[4]), distancia)
                dep.municipios.append(mun)
                deplista.append(dep)
            else:
                distancia = haversine(
                    14.589246, -90.551449, float(valor[3]), float(valor[4]))
                dep = getDepartamento(valor[0], deplista)
                mun = Municipio(valor[2], valor[1], float(
                    valor[3]), float(valor[4]), distancia)
                dep.municipios.append(mun)

        

        

        return deplista


def existeDepartamento(numero, lista):
    for valor in lista:
        if valor.numero == numero:
            return True
    return False


def getDepartamento(numero, lista):
    for elemento in lista:
        if elemento.numero == numero:
            return elemento
    return None


def haversine(lat1, lon1, lat2, lon2):
    rad = math.pi/180
    dlat = lat2-lat1
    dlon = lon2-lon1
    R = 6372.795477598
    a = (math.sin(rad*dlat/2))**2 + math.cos(rad*lat1) * \
        math.cos(rad*lat2)*(math.sin(rad*dlon/2))**2
    distancia = 2*R*math.asin(math.sqrt(a))
    return distancia


def escalarDisatancia(lista):
    maximo = getMaxDistancia(lista)
    minimo = getMinDistancia(lista)

    for elemento in lista:
        for mun in elemento.municipios:
            dis = mun.distancia
            escal = (dis - minimo) / (maximo - minimo)
            mun.distancia = escal


def getMaxDistancia(lista):
    valor = lista[0].municipios[0].distancia
    for elemento in lista:
        for mun in elemento.municipios:
            if valor < mun.distancia:
                valor = mun.distancia
    return valor


def getMinDistancia(lista):
    valor = lista[0].municipios[0].distancia
    for elemento in lista:
        for mun in elemento.municipios:
            if valor > mun.distancia:
                valor = mun.distancia
    return valor


def leerData():
    with open('archivos/Dataset.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile)
        valores = list(spamreader)
        del valores[0]
        dep_list = leerMunicipios()
        entrada = []
        salida = []
        datos = []
        # [Genereo, Edad, Año, Distancia]
        for elemento in valores:
            genero = 0
            if str(elemento[1]).lower() == "masculino":
                genero = 1
            edad = float(elemento[2])
            year = float(elemento[7])
            dep = getDepartamento(elemento[3], dep_list)
            if dep.nombre == "":
                dep.nombre = elemento[4]
            distancia = getDistancia(dep.municipios, elemento[5])
            #entrada.append([genero, edad, year, distancia])
            respuesta = 0
            if str(elemento[0]).lower() == "activo":
                respuesta = 1
            #salida.append([respuesta])
            datos.append([[genero, edad, year, distancia],[respuesta]])

        random.shuffle(datos)

        for elemento in datos:
            entrada.append(elemento[0])
            salida.append(elemento[1])

        print(entrada[0])
        # Escalonar edad
        maxEdad, minEdad = escalarPosicio(entrada, 1)
        # Escalonar año
        maxYear, minYear = escalarPosicio(entrada, 2)
        # La distancia ya esta escalonada
        maxDist, minDist = escalarPosicio(entrada, 3)
        print(entrada[0])

        datos_arr = np.asarray(entrada)
        resp_arr = np.asarray(salida)
        

        slice_point = int(datos_arr.shape[0] * 0.8)

        datos_entrenamiento = datos_arr[0:slice_point]
        respuesta_entrenamiento = resp_arr[0:slice_point]

        datos_prueba = datos_arr[slice_point:]
        respuesta_prueba = resp_arr[slice_point:]

        print(datos_entrenamiento.shape)
        print(respuesta_entrenamiento.shape)

        print(datos_prueba.shape)
        print(respuesta_prueba.shape)

        # Transpuesta
        datos_entrenamiento = datos_entrenamiento.T
        respuesta_entrenamiento = respuesta_entrenamiento.T

        datos_prueba = datos_prueba.T
        respuesta_prueba = respuesta_prueba.T

        print(datos_entrenamiento.shape)
        print(respuesta_entrenamiento.shape)

        print(datos_prueba.shape)
        print(respuesta_prueba.shape)

        return datos_entrenamiento, respuesta_entrenamiento, datos_prueba, respuesta_prueba, maxEdad, maxYear, maxDist, minEdad, minYear, minDist, dep_list
        # print(salida)


def getDistancia(lista, numero):
    for mun in lista:
        if mun.numero == numero:
            return mun.distancia
    print("*-* No se encontro el numero de municipio: ", numero)
    return None


def escalarPosicio(lista, numero):
    maximo = getMaximo(lista, numero)
    minimo = getMinimo(lista, numero)
    for elemento in lista:
        valor = elemento[numero]
        escalonado = (valor - minimo) / (maximo - minimo)
        elemento[numero] = escalonado
    return maximo, minimo



def getMaximo(lista, posicion):
    valor = lista[0][posicion]
    for elemento in lista:
        if valor < elemento[posicion]:
            valor = elemento[posicion]
    return valor


def getMinimo(lista, posicion):
    valor = lista[0][posicion]
    for elemento in lista:
        if valor > elemento[posicion]:
            valor = elemento[posicion]
    return valor


