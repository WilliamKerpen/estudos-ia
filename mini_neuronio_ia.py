# ============================================================
# MINI NEURÔNIO ARTIFICIAL EM PYTHON
# ============================================================
#
# Vamos criar um neurônio artificial do zero.
#
# O neurônio receberá 3 informações:
#
#   1. Horas estudadas
#   2. Horas dormidas
#   3. Número de exercícios realizados
#
# E tentará prever uma nota.
#
# Não vamos usar nenhuma biblioteca de Machine Learning.
#
# ============================================================


# ------------------------------------------------------------
# 1. DADOS DE TREINAMENTO
# ------------------------------------------------------------

# Cada lista representa uma característica dos alunos.

horas_estudo = [1, 2, 3, 4, 5]
horas_sono = [5, 6, 7, 7, 8]
exercicios = [1, 2, 2, 3, 4]


# Notas reais dos alunos.

notas = [2, 4, 6, 8, 10]


# ------------------------------------------------------------
# 2. PESOS DO NEURÔNIO
# ------------------------------------------------------------

# Agora temos um peso para cada entrada.
#
# w1 -> importância das horas de estudo
# w2 -> importância das horas de sono
# w3 -> importância dos exercícios
#
# Começamos todos com zero.
#
# O treinamento vai modificar esses valores.

peso_estudo = 0.0
peso_sono = 0.0
peso_exercicios = 0.0


# ------------------------------------------------------------
# 3. BIAS
# ------------------------------------------------------------

# O bias é um valor adicional que permite
# deslocar a saída do neurônio.

bias = 0.0


# ------------------------------------------------------------
# 4. TAXA DE APRENDIZADO
# ------------------------------------------------------------

# Define o tamanho do passo que será dado
# durante o aprendizado.

learning_rate = 0.001


# ------------------------------------------------------------
# 5. FUNÇÃO DO NEURÔNIO
# ------------------------------------------------------------

def neuronio(estudo, sono, exercicios):
    """
    Calcula a saída do neurônio.

    Fórmula:

        saída =
            estudo * peso_estudo
            +
            sono * peso_sono
            +
            exercicios * peso_exercicios
            +
            bias
    """

    resultado = (
        estudo * peso_estudo
        + sono * peso_sono
        + exercicios * peso_exercicios
        + bias
    )

    return resultado


# ------------------------------------------------------------
# 6. TREINAMENTO
# ------------------------------------------------------------

# Vamos passar várias vezes pelos dados.

for epoca in range(1000):

    perda_total = 0.0


    # --------------------------------------------------------
    # Percorrer todos os exemplos
    # --------------------------------------------------------

    for estudo, sono, exercicio, real in zip(
        horas_estudo,
        horas_sono,
        exercicios,
        notas
    ):

        # ----------------------------------------------------
        # 6.1 - FAZER A PREVISÃO
        # ----------------------------------------------------

        previsao = neuronio(
            estudo,
            sono,
            exercicio
        )


        # ----------------------------------------------------
        # 6.2 - CALCULAR O ERRO
        # ----------------------------------------------------

        erro = previsao - real


        # ----------------------------------------------------
        # 6.3 - CALCULAR O LOSS
        # ----------------------------------------------------

        loss = erro ** 2

        perda_total += loss


        # ----------------------------------------------------
        # 6.4 - CALCULAR OS GRADIENTES
        # ----------------------------------------------------
        #
        # Cada peso precisa descobrir quanto deve mudar.
        #
        # A lógica é parecida com o nosso exemplo anterior.
        #
        # Para cada entrada:
        #
        # gradiente = 2 * entrada * erro
        #

        gradiente_estudo = 2 * estudo * erro

        gradiente_sono = 2 * sono * erro

        gradiente_exercicios = 2 * exercicio * erro


        # O bias não possui uma entrada.
        #
        # Por isso:

        gradiente_bias = 2 * erro


        # ----------------------------------------------------
        # 6.5 - ATUALIZAR OS PESOS
        # ----------------------------------------------------

        peso_estudo = (
            peso_estudo
            - learning_rate * gradiente_estudo
        )


        peso_sono = (
            peso_sono
            - learning_rate * gradiente_sono
        )


        peso_exercicios = (
            peso_exercicios
            - learning_rate * gradiente_exercicios
        )


        # ----------------------------------------------------
        # 6.6 - ATUALIZAR O BIAS
        # ----------------------------------------------------

        bias = (
            bias
            - learning_rate * gradiente_bias
        )


    # --------------------------------------------------------
    # 6.7 - CALCULAR A PERDA MÉDIA
    # --------------------------------------------------------

    perda_media = perda_total / len(notas)


    # --------------------------------------------------------
    # MOSTRAR O PROGRESSO
    # --------------------------------------------------------

    if epoca % 100 == 0:

        print(
            f"Época: {epoca:4d} | "
            f"Loss: {perda_media:.6f}"
        )


# ============================================================
# 7. MOSTRAR O NEURÔNIO TREINADO
# ============================================================

print("\n===================================")
print("NEURÔNIO TREINADO")
print("===================================")

print(f"Peso estudo:      {peso_estudo:.6f}")
print(f"Peso sono:        {peso_sono:.6f}")
print(f"Peso exercícios:  {peso_exercicios:.6f}")
print(f"Bias:             {bias:.6f}")


# ============================================================
# 8. FAZER UMA NOVA PREVISÃO
# ============================================================

print("\n===================================")
print("NOVA PREVISÃO")
print("===================================")


# Imagine uma pessoa que:
#
#   estudou 6 horas
#   dormiu 8 horas
#   fez 4 exercícios
#
# Vamos pedir para o neurônio prever a nota.

nova_previsao = neuronio(
    6,
    8,
    4
)


print(
    f"Nota prevista: {nova_previsao:.2f}"
)


# ============================================================
# FIM
# ============================================================
