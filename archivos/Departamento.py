class Departamento:
    def __init__(self, numero):
        self.numero = numero
        self.nombre  = ""
        self.municipios = []

    def setNombre(self, nombre):
        self.nombre = nombre