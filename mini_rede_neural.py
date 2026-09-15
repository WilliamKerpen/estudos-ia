# ==========================================
# REDE NEURAL DO ZERO
# 3 entradas
# 3 neurônios na camada escondida
# 1 neurônio de saída
# ==========================================


# ------------------------------------------
# DADOS DE TREINAMENTO
# ------------------------------------------
#
# estudo, sono, exercício, nota
#
dados = [
    (1, 6, 1, 3),
    (2, 7, 1, 5),
    (3, 7, 2, 6),
    (4, 8, 2, 8),
    (5, 8, 3, 10),
]


# ------------------------------------------
# PESOS DA CAMADA ESCONDIDA
# ------------------------------------------
#
# Cada neurônio possui seus próprios pesos.
#
# N1
peso_estudo_1 = 0.10
peso_sono_1 = 0.20
peso_exercicio_1 = 0.30
bias_1 = 0.10

# N2
peso_estudo_2 = -0.20
peso_sono_2 = 0.40
peso_exercicio_2 = 0.10
bias_2 = 0.20

# N3
peso_estudo_3 = 0.30
peso_sono_3 = -0.10
peso_exercicio_3 = 0.50
bias_3 = 0.30


# ------------------------------------------
# PESOS DO NEURÔNIO DE SAÍDA
# ------------------------------------------
#
# O N4 recebe as saídas de N1, N2 e N3.
#
peso_n1_saida = 0.40
peso_n2_saida = 0.30
peso_n3_saida = 0.20
bias_saida = 0.10


# ------------------------------------------
# TAXA DE APRENDIZADO
# ------------------------------------------

learning_rate = 0.0001


# ------------------------------------------
# FUNÇÃO ReLU
# ------------------------------------------
#
# ReLU:
#
# se x > 0 → retorna x
# se x <= 0 → retorna 0
#
def relu(x):
    if x > 0:
        return x

    return 0


# ------------------------------------------
# DERIVADA DA ReLU
# ------------------------------------------

def derivada_relu(x):
    if x > 0:
        return 1

    return 0


# ------------------------------------------
# TREINAMENTO
# ------------------------------------------

for epoca in range(100000):

    for estudo, sono, exercicio, real in dados:

        # ==================================
        # FORWARD PASS
        # ==================================

        # ----------------------------------
        # NEURÔNIO 1
        # ----------------------------------

        z1 = (
            estudo * peso_estudo_1
            + sono * peso_sono_1
            + exercicio * peso_exercicio_1
            + bias_1
        )

        n1 = relu(z1)


        # ----------------------------------
        # NEURÔNIO 2
        # ----------------------------------

        z2 = (
            estudo * peso_estudo_2
            + sono * peso_sono_2
            + exercicio * peso_exercicio_2
            + bias_2
        )

        n2 = relu(z2)


        # ----------------------------------
        # NEURÔNIO 3
        # ----------------------------------

        z3 = (
            estudo * peso_estudo_3
            + sono * peso_sono_3
            + exercicio * peso_exercicio_3
            + bias_3
        )

        n3 = relu(z3)


        # ----------------------------------
        # NEURÔNIO DE SAÍDA
        # ----------------------------------
        #
        # O N4 recebe N1, N2 e N3.
        #
        previsao = (
            n1 * peso_n1_saida
            + n2 * peso_n2_saida
            + n3 * peso_n3_saida
            + bias_saida
        )


        # ==================================
        # LOSS
        # ==================================

        erro = previsao - real

        loss = erro ** 2


        # ==================================
        # BACKPROPAGATION
        # ==================================
        #
        # Primeiro calculamos o gradiente
        # do neurônio de saída.
        #
        # Loss = erro²
        #
        # dLoss/dPrevisao = 2 * erro
        #

        gradiente_saida = 2 * erro


        # ----------------------------------
        # Gradientes dos pesos da saída
        # ----------------------------------

        gradiente_peso_n1_saida = gradiente_saida * n1
        gradiente_peso_n2_saida = gradiente_saida * n2
        gradiente_peso_n3_saida = gradiente_saida * n3

        gradiente_bias_saida = gradiente_saida


        # ----------------------------------
        # Gradientes que voltam para
        # N1, N2 e N3
        # ----------------------------------

        gradiente_n1 = (
            gradiente_saida
            * peso_n1_saida
            * derivada_relu(z1)
        )

        gradiente_n2 = (
            gradiente_saida
            * peso_n2_saida
            * derivada_relu(z2)
        )

        gradiente_n3 = (
            gradiente_saida
            * peso_n3_saida
            * derivada_relu(z3)
        )


        # ----------------------------------
        # Gradientes dos pesos de N1
        # ----------------------------------

        gradiente_peso_estudo_1 = gradiente_n1 * estudo
        gradiente_peso_sono_1 = gradiente_n1 * sono
        gradiente_peso_exercicio_1 = gradiente_n1 * exercicio
        gradiente_bias_1 = gradiente_n1


        # ----------------------------------
        # Gradientes dos pesos de N2
        # ----------------------------------

        gradiente_peso_estudo_2 = gradiente_n2 * estudo
        gradiente_peso_sono_2 = gradiente_n2 * sono
        gradiente_peso_exercicio_2 = gradiente_n2 * exercicio
        gradiente_bias_2 = gradiente_n2


        # ----------------------------------
        # Gradientes dos pesos de N3
        # ----------------------------------

        gradiente_peso_estudo_3 = gradiente_n3 * estudo
        gradiente_peso_sono_3 = gradiente_n3 * sono
        gradiente_peso_exercicio_3 = gradiente_n3 * exercicio
        gradiente_bias_3 = gradiente_n3


        # ==================================
        # GRADIENT DESCENT
        # ==================================

        # ----------------------------------
        # Atualizar pesos da saída
        # ----------------------------------

        peso_n1_saida -= (
            learning_rate * gradiente_peso_n1_saida
        )

        peso_n2_saida -= (
            learning_rate * gradiente_peso_n2_saida
        )

        peso_n3_saida -= (
            learning_rate * gradiente_peso_n3_saida
        )

        bias_saida -= (
            learning_rate * gradiente_bias_saida
        )


        # ----------------------------------
        # Atualizar pesos de N1
        # ----------------------------------

        peso_estudo_1 -= (
            learning_rate * gradiente_peso_estudo_1
        )

        peso_sono_1 -= (
            learning_rate * gradiente_peso_sono_1
        )

        peso_exercicio_1 -= (
            learning_rate * gradiente_peso_exercicio_1
        )

        bias_1 -= (
            learning_rate * gradiente_bias_1
        )


        # ----------------------------------
        # Atualizar pesos de N2
        # ----------------------------------

        peso_estudo_2 -= (
            learning_rate * gradiente_peso_estudo_2
        )

        peso_sono_2 -= (
            learning_rate * gradiente_peso_sono_2
        )

        peso_exercicio_2 -= (
            learning_rate * gradiente_peso_exercicio_2
        )

        bias_2 -= (
            learning_rate * gradiente_bias_2
        )


        # ----------------------------------
        # Atualizar pesos de N3
        # ----------------------------------

        peso_estudo_3 -= (
            learning_rate * gradiente_peso_estudo_3
        )

        peso_sono_3 -= (
            learning_rate * gradiente_peso_sono_3
        )

        peso_exercicio_3 -= (
            learning_rate * gradiente_peso_exercicio_3
        )

        bias_3 -= (
            learning_rate * gradiente_bias_3
        )


# ==========================================
# MOSTRAR PESOS APRENDIDOS
# ==========================================

print("\n===== PESOS APRENDIDOS =====")


print("\nNeurônio 1:")
print("Peso estudo:", peso_estudo_1)
print("Peso sono:", peso_sono_1)
print("Peso exercício:", peso_exercicio_1)
print("Bias:", bias_1)


print("\nNeurônio 2:")
print("Peso estudo:", peso_estudo_2)
print("Peso sono:", peso_sono_2)
print("Peso exercício:", peso_exercicio_2)
print("Bias:", bias_2)


print("\nNeurônio 3:")
print("Peso estudo:", peso_estudo_3)
print("Peso sono:", peso_sono_3)
print("Peso exercício:", peso_exercicio_3)
print("Bias:", bias_3)


print("\nNeurônio de saída:")
print("Peso N1:", peso_n1_saida)
print("Peso N2:", peso_n2_saida)
print("Peso N3:", peso_n3_saida)
print("Bias:", bias_saida)


# ==========================================
# NOVA PREVISÃO
# ==========================================

estudo = 6
sono = 8
exercicio = 3


# N1
z1 = (
    estudo * peso_estudo_1
    + sono * peso_sono_1
    + exercicio * peso_exercicio_1
    + bias_1
)

n1 = relu(z1)


# N2
z2 = (
    estudo * peso_estudo_2
    + sono * peso_sono_2
    + exercicio * peso_exercicio_2
    + bias_2
)

n2 = relu(z2)


# N3
z3 = (
    estudo * peso_estudo_3
    + sono * peso_sono_3
    + exercicio * peso_exercicio_3
    + bias_3
)

n3 = relu(z3)


# N4
previsao = (
    n1 * peso_n1_saida
    + n2 * peso_n2_saida
    + n3 * peso_n3_saida
    + bias_saida
)


print("\n===== NOVA PREVISÃO =====")

print("Saída N1:", n1)
print("Saída N2:", n2)
print("Saída N3:", n3)

print("Nota prevista:", previsao)

