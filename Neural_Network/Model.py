import numpy as np
#np.set_printoptions(threshold=100000) #Esto es para que al imprimir un arreglo no me muestre puntos suspensivos


class NN_Model:

    def __init__(self, train_set, layers, alpha=0.3, iterations=300000, lambd=0, keep_prob=1):
        self.data = train_set
        self.alpha = alpha
        self.max_iteration = iterations
        self.lambd = lambd
        self.kp = keep_prob
        # Se inicializan los pesos
        self.parametros = self.Inicializar(layers)
        self.layers = layers

    def Inicializar(self, layers):
        parametros = {}
        L = len(layers)
        print('layers:', layers)
        for l in range(1, L):
            #np.random.randn(layers[l], layers[l-1])
            #Crea un arreglo que tiene layers[l] arreglos, donde cada uno de estos arreglos tiene layers[l-1] elementos con valores aleatorios
            #np.sqrt(layers[l-1] se saca la raiz cuadrada positiva de la capa anterior ---> layers[l-1]
            parametros['W'+str(l)] = np.random.randn(layers[l], layers[l-1]) / np.sqrt(layers[l-1])
            parametros['b'+str(l)] = np.zeros((layers[l], 1))
            #print(layers[l], layers[l-1], np.random.randn(layers[l], layers[l-1]))
            #print(np.sqrt(layers[l-1]))
            #print(np.random.randn(layers[l], layers[l-1]) / np.sqrt(layers[l-1]))

        return parametros

    def training(self, show_cost=False):
        self.bitacora = []
        for i in range(0, self.max_iteration):
            y_hat, temp = self.propagacion_adelante(self.data)
            cost = self.cost_function(y_hat)
            gradientes = self.propagacion_atras(temp)
            self.actualizar_parametros(gradientes)
            if i % 50 == 0:
                self.bitacora.append(cost)
                if show_cost:
                    print('Iteracion No.', i, 'Costo:', cost, sep=' ')


    
    def propagacion_adelante(self, dataSet):
        # Se extraen las entradas
        X = dataSet.x

        
        temp = {}

        for y in range(len(self.layers) - 2):
            w = self.parametros["W" + str(y + 1) ]
            b = self.parametros["b" + str(y + 1) ]
            
            Z = np.dot(w, X) + b
            A = self.activation_function('relu', Z)
            #Se aplica el Dropout Invertido
            D = None
            if y == 0:
                D = np.random.rand(A.shape[0], A.shape[1]) #Se generan número aleatorios para cada neurona
            else:
                D = np.random.rand(A.shape[0], X.shape[1])
            D = (D < self.kp).astype(int) #Mientras más alto es kp mayor la probabilidad de que la neurona permanezca
            A *= D
            A /= self.kp

            X = A
            temp["Z" + str(y + 1) ] = Z
            temp["A" + str(y + 1) ] = A
            temp["D" + str(y + 1) ] = D


        # ------ Ultima capa
        w = self.parametros["W" + str(len(self.layers) - 1) ]
        b = self.parametros["b" + str(len(self.layers) - 1) ]
        Z = np.dot(w, X) + b
        A = self.activation_function('sigmoide', Z)
        temp["Z" +  str(len(self.layers) - 1) ] = Z
        temp["A" + str(len(self.layers) - 1) ] = A
        
        return A, temp

    def propagacion_atras(self, temp):
        # Se obtienen los datos
        m = self.data.m
        Y = self.data.y
        X = self.data.x
       

        # Derivadas parciales ultima capa
        gradientes = {}

        A = temp["A"+str(len(self.layers) - 1)]
        dZ = A - Y
        A = temp["A"+str(len(self.layers) - 2)]
        W = self.parametros["W"+str(len(self.layers) - 1)]
        dW = (1 / m) * np.dot(dZ, A.T) + (self.lambd / m) * W
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        gradientes["dZ" + str(len(self.layers) - 1)] = dZ
        gradientes["dW" + str(len(self.layers) - 1)] = dW
        gradientes["db" + str(len(self.layers) - 1)] = db

        # Derivadas parciales de capas intermedias
        num = len(self.layers) - 2
        while num != 1:
            W = self.parametros["W" + str(num + 1)]
            dA = np.dot(W.T, dZ)
            D = temp["D" + str(num)]
            dA *= D
            dA /= self.kp
            A = temp["A"+ str(num)]
            dZ = np.multiply(dA, np.int64(A > 0))
            A = temp["A"+ str(num - 1)]
            W = self.parametros["W" + str(num)]
            dW = 1. / m * np.dot(dZ, A.T) + (self.lambd / m) * W
            db = 1. / m * np.sum(dZ, axis=1, keepdims=True)
            gradientes["dZ" + str(num)] = dZ
            gradientes["dW" + str(num)] = dW
            gradientes["db" + str(num)] = db
            gradientes["dA" + str(num)] = dA
            num -= 1
        
        

        # Derivadas parciales de la primera capa
        dA1 = np.dot(W.T, dZ)
        D1 = temp["D1"]
        dA1 *= D1
        dA1 /= self.kp
        A1 = temp["A1"]
        dZ1 = np.multiply(dA1, np.int64(A1 > 0))
        W1 = self.parametros["W1"]
        dW1 = 1./m * np.dot(dZ1, X.T) + (self.lambd / m) * W1
        db1 = 1./m * np.sum(dZ1, axis=1, keepdims = True)
        gradientes["dZ1"] = dZ1
        gradientes["dW1"] = dW1
        gradientes["db1"] = db1
        gradientes["dA1"] = dA1

        

        return gradientes

    def actualizar_parametros(self, grad):
        # Se obtiene la cantidad de pesos
        L = len(self.parametros) // 2
        for k in range(L):
            self.parametros["W" + str(k + 1)] -= self.alpha * grad["dW" + str(k + 1)]
            self.parametros["b" + str(k + 1)] -= self.alpha * grad["db" + str(k + 1)]

    def cost_function(self, y_hat):
        # Se obtienen los datos
        Y = self.data.y
        m = self.data.m
        # Se hacen los calculos
        temp = np.multiply(-np.log(y_hat), Y) + np.multiply(-np.log(1 - y_hat), 1 - Y)
        result = (1 / m) * np.nansum(temp)
        # Se agrega la regularizacion L2
        if self.lambd > 0:
            L = len(self.parametros) // 2
            suma = 0
            for i in range(L):
                suma += np.sum(np.square(self.parametros["W" + str(i + 1)]))
            result += (self.lambd/(2*m)) * suma
        return result

    def predict(self, dataSet):
        # Se obtienen los datos
        m = dataSet.m
        Y = dataSet.y
        p = np.zeros((1, m), dtype= np.int)
        # Propagacion hacia adelante
        y_hat, temp = self.propagacion_adelante(dataSet)
        # Convertir probabilidad
        for i in range(0, m):
            p[0, i] = 1 if y_hat[0, i] > 0.5 else 0
        exactitud = np.mean((p[0, :] == Y[0, ]))
        print("Exactitud: " + str(exactitud))
        return exactitud


    def activation_function(self, name, x):
        result = 0
        if name == 'sigmoide':
            result = 1/(1 + np.exp(-x))
        elif name == 'tanh':
            result = np.tanh(x)
        elif name == 'relu':
            result = np.maximum(0, x)
        
        #print('name:', name, 'result:', result)
        return result