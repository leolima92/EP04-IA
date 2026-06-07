# Relatório — EP Carro Autônomo com Q-Learning Tabular

## 1. Introdução

Este trabalho implementa um agente de aprendizado por reforço para controlar um carrinho em pistas 2D. O objetivo do agente é aprender a chegar da largada até a linha de chegada usando apenas informações locais do ambiente, sem conhecer o mapa completo da pista.

Diferente de uma abordagem de busca, em que o agente recebe o mapa e calcula um caminho, neste EP o carrinho aprende por tentativa e erro. A cada passo, ele observa o estado atual, escolhe uma ação, recebe uma recompensa e atualiza sua tabela Q. A solução foi implementada com Q-Learning tabular, usando discretização do estado contínuo.

O agente foi treinado nas pistas 01 a 16 e avaliado nas pistas 17 e 18, que foram mantidas como holdout. Dessa forma, a avaliação mede a capacidade de generalização do agente em pistas que ele não viu durante o treinamento.

---

## 2. Modelagem do problema

## 2.1 Estado observável

O estado físico interno do carrinho possui posição, ângulo e velocidade, mas o agente não recebe essas informações diretamente. O agente observa apenas um vetor com 6 valores normalizados:

```text
[d_0, d_+30, d_-30, d_+60, d_-60, velocidade_normalizada]
```

Os cinco primeiros valores representam sensores tipo LIDAR simulados. Cada sensor mede a distância até a parede mais próxima em uma direção relativa ao ângulo atual do carro. O sexto valor representa a velocidade atual dividida pela velocidade máxima.

Essa modelagem torna o problema mais interessante, porque o agente não sabe em qual posição da pista está. Ele precisa tomar decisões apenas com base no que “enxerga” localmente à frente e nas laterais.

## 2.2 Discretização do estado

Como o Q-Learning tabular precisa indexar uma tabela por estados discretos, o vetor contínuo de 6 floats foi transformado em uma tupla de inteiros. Para isso, foi usada discretização uniforme com `K = 5`.

A função usada foi:

```python
def discretizar(self, obs):
    return tuple(
        min(int(float(v) * self.K), self.K - 1)
        for v in obs
    )
```

Com `K = 5`, cada componente do estado fica em um dos baldes de 0 a 4. Assim, um vetor como:

```text
[0.35, 1.00, 0.30, 0.41, 0.18, 0.50]
```

pode virar a chave:

```text
(1, 4, 1, 2, 0, 2)
```

Essa chave é usada para acessar a tabela Q. A escolha de `K = 5` é adequada porque a velocidade do carro também possui 5 níveis possíveis: 0, 0.5, 1.0, 1.5 e 2.0. Além disso, mantém o espaço de estados em um tamanho viável para treinamento tabular.

## 2.3 Espaço de ações

O agente possui 5 ações possíveis:

| Código | Ação             |
| ------ | ---------------- |
| 0      | Nada             |
| 1      | Acelerar         |
| 2      | Frear            |
| 3      | Virar à esquerda |
| 4      | Virar à direita  |

A ação de acelerar aumenta a velocidade em 0.5 até o limite máximo. A ação de frear reduz a velocidade em 0.5 até o mínimo 0. As ações de virar alteram o ângulo do carro em 30 graus, sem alterar diretamente a velocidade.

Essa separação entre velocidade e direção faz com que o agente precise aprender uma política que coordene os dois fatores. Por exemplo, acelerar em uma reta pode ser bom, mas acelerar antes de uma curva pode causar colisão.

## 2.4 Função de recompensa

A função de recompensa usada no ambiente combina quatro ideias principais:

1. Recompensa positiva por progresso na pista.
2. Pequena penalidade por passo, incentivando o agente a terminar rápido.
3. Penalidade grande por colisão.
4. Bônus grande ao cruzar a linha de chegada.

A recompensa de progresso é baseada em um campo calculado por BFS a partir da largada. Assim, o agente recebe recompensa quando atinge células que representam avanço real na pista. Isso ajuda o aprendizado, pois uma recompensa apenas no final seria muito esparsa.

A colisão gera penalidade de `-100`, enquanto a chegada gera bônus de `+500`.

---

## 3. Implementação do Q-Learning

## 3.1 Estrutura da tabela Q

A tabela Q foi implementada como um dicionário Python:

```python
self.Q = {}
```

A chave do dicionário é o estado discretizado, representado por uma tupla de 6 inteiros. O valor associado é um vetor NumPy com 5 posições, uma para cada ação possível.

Quando um estado ainda não existe na tabela, ele é inicializado com valores zero:

```python
if chave not in self.Q:
    self.Q[chave] = np.zeros(self.n_actions, dtype=np.float32)
```

Essa escolha evita alocar todos os estados possíveis antecipadamente. Em vez disso, apenas os estados realmente visitados durante o treinamento são armazenados.

## 3.2 Atualização da tabela Q

A atualização implementada segue a regra clássica do Q-Learning:

```text
Q(s,a) ← Q(s,a) + α [r + γ max Q(s',a') - Q(s,a)]
```

No código, quando o próximo estado é terminal, o alvo é apenas a recompensa imediata. Caso contrário, o alvo considera a melhor ação possível no próximo estado.

Os hiperparâmetros utilizados foram:

| Hiperparâmetro                | Valor |
| ----------------------------- | ----- |
| Taxa de aprendizado `alpha`   | 0.1   |
| Fator de desconto `gamma`     | 0.99  |
| Epsilon inicial               | 1.0   |
| Epsilon final                 | 0.05  |
| Discretização `K`             | 5     |
| Máximo de passos por episódio | 500   |
| Seed                          | 42    |

A taxa de aprendizado `0.1` foi escolhida por ser um valor moderado, permitindo que a tabela Q seja atualizada gradualmente sem oscilar demais. O fator de desconto `0.99` faz sentido porque a recompensa de chegada pode estar muitos passos à frente, então o agente precisa valorizar recompensas futuras.

## 3.3 Política de exploração

Durante o treinamento, foi usada política epsilon-greedy. Com probabilidade `epsilon`, o agente escolhe uma ação aleatória. Com probabilidade `1 - epsilon`, escolhe a ação com maior valor Q para o estado atual.

O epsilon começa em `1.0`, ou seja, o agente explora bastante no início. Ao longo dos primeiros 80% dos episódios, o epsilon decai linearmente até `0.05`. Nos 20% finais, ele permanece em `0.05`, mantendo uma pequena exploração residual.

Essa estratégia foi importante porque no início a tabela Q ainda está vazia. Se o agente agisse de forma gulosa desde o começo, poderia ficar preso em uma política ruim sem explorar alternativas.

## 3.4 Treinamento round-robin

O treinamento foi feito usando as pistas 01 a 16. A cada episódio, uma pista de treino era sorteada aleatoriamente. Essa estratégia foi usada para evitar que o agente treinasse demais em uma pista específica e esquecesse comportamentos úteis aprendidos em outras.

O conjunto de treino foi:

```text
pistas/pista_01.txt até pistas/pista_16.txt
```

As pistas 17 e 18 não foram usadas no treinamento. Elas foram reservadas apenas para avaliação final, como pistas de holdout.

O treinamento usado nesta versão teve:

```text
1000 episódios por pista × 16 pistas = 16000 episódios totais
```

Ao final do treinamento, a tabela Q populou 3510 estados discretizados.

---

## 4. Resultados

## 4.1 Evolução durante o treinamento

Durante o treinamento com 16000 episódios totais, o agente começou com desempenho ruim, recebendo recompensas médias próximas de `-95` e taxa de sucesso de `0%` nas janelas iniciais. Isso indica que, no começo, o comportamento era quase totalmente exploratório e resultava principalmente em colisões.

Com o avanço do treinamento e a redução do epsilon, o desempenho começou a melhorar. No final, a média de recompensa ficou bem menos negativa e houve janelas com taxa de sucesso chegando a aproximadamente `12%`. Isso mostra que o agente aprendeu alguns padrões úteis de navegação, mesmo que ainda não tenha convergido para uma política perfeita.

Resumo do treinamento:

| Métrica                                                     | Valor |
| ----------------------------------------------------------- | ----- |
| Episódios totais                                            | 16000 |
| Estados populados                                           | 3510  |
| Epsilon final                                               | 0.05  |
| Melhor taxa de sucesso observada em janela de 100 episódios | 12%   |

## 4.2 Resultados nas pistas de holdout

As pistas 17 e 18 foram usadas apenas para avaliação. O agente foi avaliado com política gulosa, ou seja, com `epsilon = 0`.

### Pista 17

| Métrica                    | Valor      |
| -------------------------- | ---------- |
| Sucesso                    | SIM        |
| Tempo de chegada           | 206 passos |
| Velocidade média           | 0.71       |
| Velocidade máxima atingida | 1.50       |
| Recompensa total           | 618.40     |

Na pista 17, o agente conseguiu chegar ao final mesmo sem ter treinado nela. Esse é o resultado mais importante do trabalho, pois mostra que parte da política aprendida nas pistas 01 a 16 conseguiu generalizar para uma pista nova.

A velocidade média de `0.71` indica que o agente adotou uma política relativamente conservadora. Ele não andou sempre na velocidade máxima, mas conseguiu controlar a velocidade o suficiente para completar a pista.

### Pista 18

| Métrica                    | Valor      |
| -------------------------- | ---------- |
| Sucesso                    | NAO        |
| Tempo                      | 500 passos |
| Velocidade média           | 0.13       |
| Velocidade máxima atingida | 1.50       |
| Recompensa total           | 10.00      |

Na pista 18, o agente não conseguiu chegar dentro do limite de 500 passos. A velocidade média muito baixa indica que a política ficou conservadora demais. O agente evitou colisões e ainda conseguiu algum progresso, mas não foi agressivo o suficiente para completar a pista no tempo disponível.

## 4.3 Análise crítica

O resultado mostra uma generalização parcial. O agente conseguiu completar a pista 17, que não fazia parte do treinamento, indicando que a representação baseada em LIDAR foi suficiente para transferir alguns comportamentos aprendidos. Por exemplo, o agente aprendeu a reagir a paredes próximas, controlar a velocidade em curvas e tomar decisões locais sem conhecer a posição absoluta na pista.

Por outro lado, a falha na pista 18 mostra uma limitação do Q-Learning tabular com estado local. Como o agente não conhece o mapa completo e não possui memória, ele toma decisões apenas com base no que os sensores mostram no momento. Em pistas mais longas ou com combinações mais complexas de curvas, essa visão local pode não ser suficiente para produzir uma política eficiente.

Além disso, a pista 18 parece ter exigido uma política mais agressiva em algumas retas e mais cuidadosa em regiões estreitas. O agente treinado aprendeu uma política segura, mas lenta. Isso fica evidente pela velocidade média de apenas `0.13` na pista 18. Portanto, o agente não necessariamente bateu por ser imprudente; ele falhou principalmente por não completar a pista dentro do limite de passos.

Mesmo assim, o resultado é coerente com a proposta do EP: o agente aprendeu por tentativa e erro, populou uma tabela Q com milhares de estados e conseguiu generalizar para pelo menos uma pista holdout. A diferença entre a pista 17 e a pista 18 também ajuda a demonstrar os limites da abordagem tabular e da representação local por sensores LIDAR.

---

## 5. Arquivos gerados

A solução gera os seguintes arquivos principais:

```text
treinamento/qlearning.pkl
q_learning_pista_17.txt
q_learning_pista_18.txt
```

O arquivo `qlearning.pkl` contém a tabela Q treinada, a discretização usada, os hiperparâmetros, a seed, o histórico de recompensas e a lista de pistas usadas no treinamento.

Os arquivos `q_learning_pista_17.txt` e `q_learning_pista_18.txt` registram as métricas finais de avaliação nas pistas holdout.

---

## 6. Como executar

Para instalar as dependências:

```bash
python -m pip install -r requirements.txt
```

Para validar as pistas:

```bash
python tests/validar_pistas.py
```

Para treinar novamente e avaliar nas pistas holdout:

```bash
python solucao.py --episodios-por-pista 1000 --recarregar
```

Para apenas carregar o modelo salvo e avaliar:

```bash
python solucao.py
```

Para avaliar uma pista específica:

```bash
python solucao.py --avaliar pistas/pista_17.txt
```

---

## 7. Conclusão

O Q-Learning tabular implementado conseguiu aprender uma política funcional a partir de sensores locais e recompensas de progresso. A solução não usa bibliotecas prontas de aprendizado por reforço; a discretização, a tabela Q, a política epsilon-greedy, o treinamento round-robin e a avaliação foram implementados diretamente.

O principal resultado foi o sucesso na pista 17, uma pista não vista durante o treinamento. Isso indica que a política aprendeu padrões locais úteis, como controlar a velocidade, reagir a paredes próximas e realizar curvas sem conhecer o mapa global.

A falha na pista 18 mostra que ainda há espaço para melhoria. Um treinamento maior, uma estratégia de exploração mais longa ou ajustes na função de recompensa poderiam melhorar o desempenho. Ainda assim, o resultado obtido já demonstra aprendizado, generalização parcial e uma análise clara das limitações da abordagem.
