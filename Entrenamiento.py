from Util import Plotter
from Neural_Network.Data import Data
from Neural_Network.Model import NN_Model
from archivos.leerDatos import leerData
from datetime import datetime
from archivos.Nodo import Nodo
from  archivos.Datos import Datos
import random
import pickle


def principal():
    datos_entrenamiento, respuesta_entrenamiento, datos_prueba, respuesta_prueba, maxEdad, maxYear, maxDist, minEdad, minYear, minDist, dep_list= leerData()

    train_set = Data(datos_entrenamiento, respuesta_entrenamiento)
    test_set = Data(datos_prueba, respuesta_prueba)

    capas = [train_set.n, 8, 6, 5, 1]

    modelo = NN_Model(train_set, capas, alpha=0.01, iterations=10000, lambd=150, keep_prob=0.85)
    
    now = datetime.now()
    fecha_hora = now.strftime("%d/%m/%Y %H:%M:%S")
    print(fecha_hora)

    modelo.training(True)
    now = datetime.now()
    fecha_hora = now.strftime("%d/%m/%Y %H:%M:%S")
    print(fecha_hora)
    print("alpa", modelo.alpha, "lam", modelo.lambd, "kp", modelo.kp, "it", modelo.max_iteration)
    #Plotter.show_Model([modelo])

    print('Entrenamiento Modelo')
    modelo.predict(train_set)
    print('Validacion Modelo')
    modelo.predict(test_set)


tamPoblacion = 10
generacionMaxima = 100
cantidadPadres = 5


def inicializarPoblacion(train_data, test_data):
    poblacion = []
    for _ in range(tamPoblacion):
        solucion = []
        for _ in range(4):
            solucion.append(random.randint(0, 9))
        fitness, modelo = evaluarFitness(solucion, train_data, test_data)
        nod = Nodo(solucion, fitness)
        nod.modelo = modelo
        poblacion.append(nod)
    return poblacion


def verificarCriterio(generacion):
    return False if generacion < generacionMaxima else True


def evaluarFitness(solucion, train_data, test_data):
    # Soucion [alpha, lambda, itereaciones, kp]
    val_alpha = [0.1, 0.01, 0.5, 0.05, 0.001,
                 0.0001, 0.009, 0.0005, 0.09, 0.007]
    val_lambda = [0, 0.1, 0.01, 0.5, 0.09, 1, 50, 300, 150, 0.001]
    val_iteraciones = [5000, 1000, 12000, 20000,
                       25000, 7500, 500, 40000, 50000, 35000]
    val_kp = [1, 0.95, 0.85, 0.65, 0.45, 0.55, 0.75, 0.25, 0.35, 0.1]
    capas = [train_data.n, 8, 6, 5, 1]
    modelo = NN_Model(train_data, capas, alpha=val_alpha[solucion[0]], iterations=val_iteraciones[
                      solucion[2]], lambd=val_lambda[solucion[1]], keep_prob=val_kp[solucion[3]])
    modelo.training(False)
    valor_Fit = modelo.predict(test_data)
    return valor_Fit, modelo


def seleccionarPadres(poblacion):
    padres = []
    poblacion = sorted(poblacion, key=lambda item: item.fitness, reverse=True)[
        :len(poblacion)]
    for x in range(cantidadPadres):
        padres.append(poblacion[x])
    return padres


def cruzar(padre1, padre2):
    hijo = []
    for x in range(len(padre1)):
        probabilidad = random.randrange(2)
        if probabilidad == 1:
            hijo.append(padre1[x])
        else:
            hijo.append(padre2[x])
    return hijo


def mutar(solucion):
    for x in range(len(solucion)):
        if random.randrange(2) == 1:
            solucion[x] = random.randint(0, 9)
    return solucion


def emparejar(padres, train_data, test_data):
    nuevaPoblacion = []
    solHijos = []

    
    hijo1 = cruzar(padres[0].solucion, padres[1].solucion)
    solHijos.append(hijo1)
    hijo2 = cruzar(padres[1].solucion, padres[2].solucion)
    solHijos.append(hijo2)
    hijo3 = cruzar(padres[2].solucion, padres[3].solucion)
    solHijos.append(hijo3)
    hijo4 = cruzar(padres[3].solucion, padres[4].solucion)
    solHijos.append(hijo4)
    hijo5 = cruzar(padres[0].solucion, padres[2].solucion)
    solHijos.append(hijo5)

    for x in range(len(solHijos)):
        if random.randrange(2) == 1:
            solHijos[x] = mutar(solHijos[x])

    hijos = []
    for elemento in solHijos:
        fitness, modelo =  evaluarFitness(elemento, train_data, test_data)
        nod = Nodo(elemento, fitness)
        nod.modelo = modelo
        hijos.append(nod)

    for x in range(len(hijos)):
        nuevaPoblacion.append(padres[x])
        nuevaPoblacion.append(hijos[x])

    return nuevaPoblacion


def imprimirPoblacion(poblacion):
    print("------------------ Poblacion -------------------")
    for individuo in poblacion:
        print("Solucion: ", individuo.solucion, "Fitness: ", individuo.fitness)


def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def cargar(filename):
    with open(filename, 'rb') as input:
        obj = pickle.load(input)
    return obj


def ejecutar():
    datos_entrenamiento, respuesta_entrenamiento, datos_prueba, respuesta_prueba, maxEdad, maxYear, maxDist, minEdad, minYear, minDist, dep_list = leerData()
    
    datos = Datos( maxEdad, maxYear, maxDist, minEdad, minYear, minDist, dep_list)
    save_object(datos, "Guardar/Datos.pkl")
    train_set = Data(datos_entrenamiento, respuesta_entrenamiento)
    test_set = Data(datos_prueba, respuesta_prueba)

    w = open("bitacora.txt", "a")
    now = datetime.now()
    fecha_hora = now.strftime("%d/%m/%Y %H:%M:%S")
    w.write("    Inicio: "+fecha_hora + "\n")
    w.close()

    generacion = 0
    p = inicializarPoblacion(train_set, test_set)
    print("------------------------- Poblacion Inicial -----------------------")
    imprimirPoblacion(p)
    fin = verificarCriterio(generacion)

    while (not fin):
        padres = seleccionarPadres(p)
        p = emparejar(padres, train_set, test_set)
        generacion += 1
        fin = verificarCriterio(generacion)
        print("---------- Generacion ", generacion, " ------------------ ")

    p = sorted(p, key=lambda item: item.fitness, reverse=True)[:len(p)]
    save_object(p[0], "Guardar/Mejor.pkl")
    save_object(p, "Guardar/PoblacionFinal.pkl")
    imprimirPoblacion(p)

    w = open("bitacora.txt", "a")
    now = datetime.now()
    fecha_hora = now.strftime("%d/%m/%Y %H:%M:%S")
    w.write("    Finalizo: "+fecha_hora + "\n")
    w.close()

    
principal()