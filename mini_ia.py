# ============================================================
# MINI IA EM PYTHON - DO ZERO
# ============================================================
#
# Objetivo:
# Ensinar uma pequena "IA" a descobrir a relação entre:
#
#     horas estudadas -> nota
#
# Dados:
#
#     1 hora  -> 2
#     2 horas -> 4
#     3 horas -> 6
#     4 horas -> 8
#     5 horas -> 10
#
# A ideia é que o modelo descubra sozinho que:
#
#     nota ≈ 2 * horas
#
# Não irei usar bibliotecas de Machine Learning.
# Tudo será feito manualmente para entender o funcionamento.
# ============================================================


# ------------------------------------------------------------
# 1. DADOS DE TREINAMENTO
# ------------------------------------------------------------

# Entradas do nosso modelo.
# Representam quantas horas a pessoa estudou.
horas = [1, 2, 3, 4, 5]

# Respostas corretas.
# Representam as notas correspondentes.
notas = [2, 4, 6, 8, 10]


# ------------------------------------------------------------
# 2. PARÂMETROS DO MODELO
# ------------------------------------------------------------

# O "peso" determina quanto a entrada influencia
# na previsão.
#
# Começamos com 0 porque queremos que o modelo
# aprenda esse valor sozinho.
peso = 0.0


# O "bias" é um valor adicional.
#
# Ele permite que o modelo desloque a previsão
# para cima ou para baixo.
bias = 0.0


# ------------------------------------------------------------
# 3. TAXA DE APRENDIZADO
# ------------------------------------------------------------

# A learning_rate determina o tamanho do passo
# que o modelo dará ao ajustar seus parâmetros.
#
# Valor pequeno:
#     aprendizado mais lento
#
# Valor grande:
#     aprendizado mais rápido, mas pode passar
#     do ponto ideal.
learning_rate = 0.01


# ------------------------------------------------------------
# 4. FUNÇÃO DE PREVISÃO
# ------------------------------------------------------------

def prever(x):
    """
    Faz uma previsão usando:

        previsão = peso * entrada + bias
    """

    return peso * x + bias


# ------------------------------------------------------------
# 5. TREINAMENTO
# ------------------------------------------------------------

# Vamos repetir o treinamento várias vezes.
#
# Cada repetição completa pelos dados é chamada
# de "época" (epoch).

for epoca in range(100):

    # Guarda a soma dos erros desta época.
    perda_total = 0.0


    # --------------------------------------------------------
    # Passamos por cada exemplo de treinamento.
    #
    # zip() junta:
    #
    # horas = [1, 2, 3, 4, 5]
    # notas = [2, 4, 6, 8, 10]
    #
    # formando:
    #
    # 1 -> 2
    # 2 -> 4
    # 3 -> 6
    # 4 -> 8
    # 5 -> 10
    # --------------------------------------------------------

    for x, real in zip(horas, notas):

        # ----------------------------------------------------
        # 5.1 - FAZER UMA PREVISÃO
        # ----------------------------------------------------

        previsao = prever(x)


        # ----------------------------------------------------
        # 5.2 - CALCULAR O ERRO
        # ----------------------------------------------------

        # Erro simples:
        #
        # previsão - valor_real
        #
        # Se for negativo:
        #     o modelo previu abaixo do correto.
        #
        # Se for positivo:
        #     o modelo previu acima do correto.

        erro = previsao - real


        # ----------------------------------------------------
        # 5.3 - CALCULAR O LOSS
        # ----------------------------------------------------

        # Elevamos o erro ao quadrado.
        #
        # Isso transforma:
        #
        # -5 -> 25
        # +5 -> 25
        #
        # Assim não temos erros positivos e negativos
        # cancelando uns aos outros.

        loss = erro ** 2


        # Adicionamos o loss ao total da época.

        perda_total += loss


        # ----------------------------------------------------
        # 5.4 - CALCULAR OS GRADIENTES
        # ----------------------------------------------------

        # O gradiente indica como devemos alterar
        # os parâmetros para diminuir o erro.
        #
        # Para:
        #
        #     previsão = peso * x + bias
        #
        # e:
        #
        #     loss = (previsão - real)²
        #
        # temos:

        gradiente_peso = 2 * x * erro

        gradiente_bias = 2 * erro


        # ----------------------------------------------------
        # 5.5 - ATUALIZAR O PESO
        # ----------------------------------------------------

        # O modelo modifica o peso na direção
        # que reduz o erro.

        peso = peso - learning_rate * gradiente_peso


        # ----------------------------------------------------
        # 5.6 - ATUALIZAR O BIAS
        # ----------------------------------------------------

        # Fazemos a mesma coisa com o bias.

        bias = bias - learning_rate * gradiente_bias


    # --------------------------------------------------------
    # 5.7 - CALCULAR A PERDA MÉDIA
    # --------------------------------------------------------

    # Temos 5 exemplos.
    #
    # Dividimos a perda total pela quantidade
    # de exemplos.

    perda_media = perda_total / len(horas)


    # --------------------------------------------------------
    # MOSTRAR O PROGRESSO
    # --------------------------------------------------------

    # Mostramos apenas algumas épocas para não
    # deixar o terminal enorme.

    if epoca % 10 == 0:

        print(
            f"Época: {epoca:3d} | "
            f"Loss: {perda_media:.6f} | "
            f"Peso: {peso:.6f} | "
            f"Bias: {bias:.6f}"
        )


# ============================================================
# 6. MODELO TREINADO
# ============================================================

print("\n==============================")
print("TREINAMENTO FINALIZADO")
print("==============================")

print(f"Peso final: {peso:.6f}")
print(f"Bias final: {bias:.6f}")


# ============================================================
# 7. FAZER NOVAS PREVISÕES
# ============================================================

print("\n==============================")
print("NOVAS PREVISÕES")
print("==============================")


# O modelo nunca recebeu estes valores durante
# o treinamento.
#
# Agora vamos perguntar quanto ele prevê
# para 6 e 7 horas de estudo.

horas_6 = prever(6)
horas_7 = prever(7)


print(f"6 horas de estudo -> nota prevista: {horas_6:.2f}")

print(f"7 horas de estudo -> nota prevista: {horas_7:.2f}")


# ============================================================
# FIM
# ============================================================
