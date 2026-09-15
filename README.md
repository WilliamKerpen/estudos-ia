# estudos-ia
repositorio para estudos de IA e Aprendizados de maquina

Mini IA & Mini Neurônio Artificial em Python

com uso de Aprendizado Supervisionado (Supervised Learning) e treinamento em Descida do Gradiente (Gradient Descent).

ste repositório contém dois projetos educacionais fundamentais para entender como funciona uma Inteligência Artificial por dentro, sem usar bibliotecas como TensorFlow, PyTorch ou scikit‑learn.

Os arquivos são:

mini_ia.py → modelo linear simples com 1 entrada e 1 peso

mini_neuronio_ia.py → neurônio artificial com 3 entradas e 3 pesos

Ambos foram desenvolvidos por William Kerpen para o curso de ADS — Faculdade FECAF, utilizando uma abordagem didática para compreender os blocos fundamentais do aprendizado de máquina.

🎯 Objetivo Geral
Ensinar, passo a passo, como funciona o processo:

Code
dados → previsão → erro → loss → gradiente → atualização → repetição
Esse ciclo é a base de praticamente todo algoritmo de Machine Learning moderno, incluindo redes neurais profundas.

🧠 Parte 1 — mini_ia.py
“A menor IA possível: 1 entrada, 1 peso”
Este script implementa um modelo linear extremamente simples:

Code
nota ≈ peso * horas + bias
📌 O que ele aprende?
Com base nos dados:

Horas	Nota
1	2
2	4
3	6
4	8
5	10


O modelo deve descobrir sozinho que:

Code
peso ≈ 2
bias ≈ 0
🔍 Conceitos envolvidos
Modelo linear

Erro

Loss MSE

Gradiente

Gradiente descendente

Épocas

📈 O que observar no terminal
Durante o treinamento:

Loss deve diminuir

peso deve se aproximar de 2

bias deve se aproximar de 0

Exemplo esperado:

Code
Peso final: 2.00
Bias final: 0.00
6 horas → 12.00
7 horas → 14.00
🧠 Parte 2 — mini_neuronio_ia.py
“Criando um neurônio artificial completo: 3 entradas, 3 pesos”
Este script evolui o conceito anterior para um neurônio artificial, com múltiplas entradas:

Code
horas estudadas
horas dormidas
exercícios realizados
Cada entrada possui seu próprio peso:

Code
w1 → peso_estudo
w2 → peso_sono
w3 → peso_exercicios
E o neurônio calcula:

Code
saída = x1*w1 + x2*w2 + x3*w3 + bias
📌 O que este neurônio aprende?
Com base nos dados:

Estudo	Sono	Exercícios	Nota
1	5	1	2
2	6	2	4
3	7	2	6
4	7	3	8
5	8	4	10


O neurônio deve descobrir sozinho quanto cada entrada influencia a nota.

Exemplo hipotético após treinamento:

Code
peso_estudo     = 1.5
peso_sono       = 0.2
peso_exercícios = 0.8
bias            = 0.1
Interpretação:

estudar tem maior impacto na nota

exercícios têm impacto moderado

sono tem impacto pequeno

bias ajusta a saída para cima/baixo

🔍 Conceitos envolvidos
Neurônio artificial

Pesos múltiplos

Bias

Função linear

Gradientes por entrada

📈 O que observar no terminal
A cada 100 épocas:

Code
Época:  0 | Loss: ...
Época:100 | Loss: ...
Época:200 | Loss: ...
...
Ao final:

Code
Peso estudo:      X
Peso sono:        Y
Peso exercícios:  Z
Bias:             B
E uma previsão para:

Code
estudo = 6
sono = 8
exercícios = 4
🔮 Conexão entre os dois scripts

mini_ia.py	    mini_neuronio_ia.py
1 entrada	    3 entradas
1 peso	        3 pesos
modelo linear	neurônio artificial
sem ativação	pronto para ativação
base do ML	    base das redes neurais


O segundo script é literalmente a evolução natural do primeiro.


### mini_rede_neural

o N4 (neurônio de saída) pega exatamente as saídas dos neurônios N1, N2 e N3, combina essas informações e produz a previsão final da rede neural.

O que é a função ReLU?
A ReLU (Rectified Linear Unit) é uma função de ativação usada em redes neurais.

Ela faz o seguinte:

Code
se x > 0 → retorna x
se x <= 0 → retorna 0
Ou seja:

A ReLU deixa passar valores positivos e zera valores negativos.

🧠 Por que isso é importante?
Porque sem uma função de ativação, uma rede neural vira apenas uma grande equação linear, incapaz de aprender relações complexas.

A ReLU introduz não linearidade, permitindo que a rede:

aprenda padrões mais complexos

combine neurônios para formar “curvas”

represente funções que não são linhas retas

Ela é usada em praticamente todas as redes modernas.

### Backpropagation

é o algoritmo que permite uma rede neural aprender ajustando seus pesos passo a passo. Backpropagation calcula como cada peso influenciou o erro e ajusta esses pesos para diminuir esse erro.

gradiente_n1 = gradiente_saida * peso_n1_saida * derivada_relu(z1)

gradiente_saida → quanto a saída errou
peso_n1_saida → quanto N1 influenciou a saída
derivada_relu(z1) → N1 estava ativo? Se não estava, não aprende