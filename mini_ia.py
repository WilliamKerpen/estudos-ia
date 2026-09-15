# dados de treinamento

horas = [1, 2, 3, 4, 5]
notas = [2, 4, 6, 8, 10]

# parametros do modelo
peso = 0.0
bias = 0.0

# previsao
def prever (x):
    return peso * x + bias

#teste inicial
print(prever(3))