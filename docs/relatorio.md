# Relatório — EP Carro Autônomo com Q-Learning Tabular

## 1. Introdução

Neste trabalho, implementamos um agente de aprendizado por reforço para controlar um carrinho em pistas 2D. O objetivo do agente é sair da largada e chegar até a linha de chegada aprendendo por tentativa e erro, sem receber o mapa completo da pista como entrada.

A solução foi feita com Q-Learning tabular. Como o estado observado pelo agente é contínuo, foi necessário discretizar as leituras dos sensores antes de usar a tabela Q. O agente recebe apenas informações locais do ambiente, como as distâncias até as paredes próximas e a velocidade atual do carro.

O treinamento foi realizado nas pistas 01 a 16. As pistas 17 e 18 foram deixadas apenas para avaliação final, funcionando como holdout. Com isso, conseguimos observar não só se o agente aprendeu nas pistas de treino, mas também se ele conseguiu generalizar parte do comportamento para pistas que não foram vistas durante o treinamento.

---

## 2. Modelagem do problema

## 2.1 Estado observável

O estado físico interno do carrinho possui posição, ângulo e velocidade, mas o agente não recebe diretamente a posição nem o ângulo. O que ele observa é um vetor com 6 valores normalizados:

```text
[d_0, d_+30, d_-30, d_+60, d_-60, velocidade_normalizada]
```

Os cinco primeiros valores são leituras dos sensores tipo LIDAR simulados. Cada sensor mede a distância até a parede mais próxima em uma direção relativa ao ângulo atual do carro. O último valor é a velocidade do carro normalizada pela velocidade máxima.

Essa escolha deixa o problema mais difícil, porque o agente não sabe exatamente onde está na pista. Ele precisa decidir apenas com base no que “enxerga” localmente. Ao mesmo tempo, isso também ajuda a política a ser mais geral, pois uma curva parecida pode gerar leituras parecidas mesmo aparecendo em outra pista.

## 2.2 Discretização do estado

Como o Q-Learning tabular precisa de estados discretos, o vetor de 6 floats foi convertido em uma tupla de inteiros. Para isso, usamos discretização uniforme com `K = 5`.

A função implementada foi:

```python
def discretizar(self, obs):
    return tuple(
        min(int(float(v) * self.K), self.K - 1)
        for v in obs
    )
```

Com `K = 5`, cada componente do estado fica em um dos baldes de 0 a 4. Por exemplo, um vetor como:

```text
[0.35, 1.00, 0.30, 0.41, 0.18, 0.50]
```

pode virar a chave:

```text
(1, 4, 1, 2, 0, 2)
```

Essa chave é usada para acessar a tabela Q.

A escolha de `K = 5` foi mantida porque combina bem com o problema. A velocidade do carro também possui 5 níveis possíveis: 0, 0.5, 1.0, 1.5 e 2.0. Além disso, esse valor mantém o espaço de estados em um tamanho viável para uma solução tabular, sem deixar a tabela grande demais para o número de episódios treinados.

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

Essa separação entre velocidade e direção é importante, porque o agente precisa aprender a coordenar as duas coisas. Em uma reta, acelerar pode ser uma boa escolha. Perto de uma curva ou parede, continuar acelerando pode levar à colisão. Por isso, a velocidade é uma parte essencial do estado.

## 2.4 Função de recompensa

A função de recompensa usada no ambiente combina quatro ideias principais:

1. recompensa positiva por progresso na pista;
2. pequena penalidade por passo;
3. penalidade grande por colisão;
4. bônus grande ao cruzar a linha de chegada.

O progresso é calculado com base em um campo de distâncias gerado por BFS a partir da largada. Assim, o agente recebe recompensa quando atinge células que representam avanço real na pista. Isso facilita o aprendizado, porque uma recompensa apenas na chegada seria muito esparsa.

A colisão gera uma penalidade de `-100`, enquanto cruzar a chegada gera um bônus de `+500`. Também existe uma penalidade pequena por tempo, de `-0.1` por passo, para incentivar o agente a terminar a pista mais rápido e não ficar parado ou girando sem objetivo.

---

## 3. Implementação do Q-Learning

## 3.1 Estrutura da tabela Q

A tabela Q foi implementada como um dicionário Python:

```python
self.Q = {}
```

A chave do dicionário é o estado discretizado, representado por uma tupla de 6 inteiros. O valor associado é um vetor NumPy com 5 posições, uma para cada ação possível.

Quando um estado ainda não existe na tabela, ele é criado com valores zerados:

```python
if chave not in self.Q:
    self.Q[chave] = np.zeros(self.n_actions, dtype=np.float32)
```

Essa estrutura foi escolhida porque nem todos os estados possíveis aparecem durante o treinamento. Assim, a tabela só guarda os estados realmente visitados, economizando memória e permitindo medir quantos estados foram populados ao final do treino.

## 3.2 Atualização da tabela Q

A atualização segue a regra clássica do Q-Learning:

```text
Q(s,a) ← Q(s,a) + α [r + γ max Q(s',a') - Q(s,a)]
```

No código, quando o próximo estado é terminal, o alvo da atualização é apenas a recompensa recebida. Quando não é terminal, o alvo considera a melhor ação possível no próximo estado.

Os hiperparâmetros usados foram:

| Hiperparâmetro                | Valor |
| ----------------------------- | ----- |
| Taxa de aprendizado `alpha`   | 0.1   |
| Fator de desconto `gamma`     | 0.99  |
| Epsilon inicial               | 1.0   |
| Epsilon final                 | 0.05  |
| Discretização `K`             | 5     |
| Máximo de passos por episódio | 500   |
| Seed                          | 42    |

A taxa de aprendizado `0.1` foi usada por ser um valor moderado. Ela permite que o agente aprenda com as novas experiências, mas sem mudar a tabela de forma brusca demais a cada passo.

O fator de desconto `0.99` foi escolhido porque a recompensa de chegada pode estar muitos passos à frente. Como a pista pode ser longa, o agente precisa valorizar recompensas futuras, não apenas o ganho imediato de cada ação.

## 3.3 Política de exploração

Durante o treinamento, usamos política epsilon-greedy. Com probabilidade `epsilon`, o agente escolhe uma ação aleatória. Com probabilidade `1 - epsilon`, ele escolhe a ação com maior valor Q para o estado atual.

No início do treinamento, `epsilon = 1.0`, ou seja, o comportamento é praticamente exploratório. Isso é importante porque a tabela Q começa vazia. Se o agente fosse guloso desde o começo, ele poderia ficar preso em escolhas ruins muito cedo.

O epsilon decai linearmente até `0.05` ao longo dos primeiros 80% dos episódios. Depois disso, permanece em `0.05` até o final. Mantivemos essa exploração residual para evitar que o agente pare completamente de testar ações alternativas.

## 3.4 Treinamento round-robin

O treinamento foi feito com as pistas 01 a 16. A cada episódio, uma pista de treino era sorteada aleatoriamente. Essa estratégia foi usada para evitar que o agente aprendesse uma pista específica e depois esquecesse comportamentos úteis ao treinar em outra.

O conjunto de treino foi:

```text
pistas/pista_01.txt até pistas/pista_16.txt
```

As pistas 17 e 18 não foram usadas no treinamento. Elas ficaram separadas apenas para a avaliação final.

O treinamento usado nesta versão teve:

```text
5000 episódios por pista × 16 pistas = 80000 episódios totais
```

Ao final do treinamento, a tabela Q populou 5065 estados discretizados.

## 3.5 Salvamento e carregamento do modelo

Para evitar treinar tudo novamente a cada execução, o modelo foi salvo em:

```text
treinamento/qlearning.pkl
```

Esse arquivo guarda a tabela Q, o valor de `K`, os hiperparâmetros, a seed, o histórico de recompensas e a lista das pistas usadas no treinamento.

A função `treinar_ou_carregar` verifica se o arquivo já existe. Se existir e o parâmetro `--recarregar` não for usado, o modelo é carregado direto do pickle. Caso contrário, o treinamento é executado novamente e o novo modelo é salvo.

---

## 4. Resultados

## 4.1 Evolução durante o treinamento

Durante o treinamento com 80000 episódios totais, o agente começou com desempenho ruim. Nas primeiras janelas, a recompensa média ficava próxima de `-96`, e a taxa de sucesso era `0%`. Isso era esperado, porque no começo o epsilon ainda estava muito alto, e o agente explorava muitas ações aleatórias.

Com o passar dos episódios, o epsilon foi diminuindo, e o desempenho começou a melhorar. A partir da segunda metade do treinamento, começaram a aparecer janelas com sucesso maior e recompensas médias menos negativas. No final, o agente já apresentava recompensas médias positivas em várias janelas.

A melhor taxa de sucesso observada em uma janela de 100 episódios foi de aproximadamente `33%`. No último registro do treinamento, a taxa de sucesso da janela estava em `30%`, com recompensa média de `124.72`.

Resumo do treinamento:

| Métrica                                                     | Valor  |
| ----------------------------------------------------------- | ------ |
| Episódios totais                                            | 80000  |
| Estados populados                                           | 5065   |
| Epsilon final                                               | 0.05   |
| Melhor taxa de sucesso observada em janela de 100 episódios | 33%    |
| Taxa de sucesso na última janela registrada                 | 30%    |
| Recompensa média na última janela registrada                | 124.72 |

Esses números mostram que o agente realmente saiu de um comportamento quase aleatório, com muitas colisões, para uma política que consegue completar parte das pistas e acumular recompensa positiva.

## 4.2 Resultados nas pistas de holdout

As pistas 17 e 18 foram usadas apenas na avaliação final. O agente foi avaliado com política gulosa, ou seja, com `epsilon = 0`.

### Pista 17

| Métrica                    | Valor      |
| -------------------------- | ---------- |
| Sucesso                    | SIM        |
| Tempo de chegada           | 164 passos |
| Velocidade média           | 0.92       |
| Velocidade máxima atingida | 1.50       |
| Recompensa total           | 623.60     |

Na pista 17, o agente conseguiu chegar ao final mesmo sem ter treinado nela. Esse foi o melhor resultado da avaliação, porque mostra que a política aprendida nas pistas 01 a 16 conseguiu generalizar para uma pista nova.

O agente completou a pista em 164 passos, com recompensa total positiva de 623.60. A velocidade média de 0.92 indica que ele não ficou parado nem excessivamente lento, mas também não correu o tempo todo na velocidade máxima. Isso sugere uma política relativamente cuidadosa, mas funcional.

### Pista 18

| Métrica                    | Valor      |
| -------------------------- | ---------- |
| Sucesso                    | NAO        |
| Tempo                      | 500 passos |
| Velocidade média           | 0.34       |
| Velocidade máxima atingida | 1.50       |
| Recompensa total           | -29.00     |

Na pista 18, o agente não conseguiu chegar dentro do limite de 500 passos. Ainda assim, ele não teve um comportamento completamente aleatório, pois conseguiu avançar durante parte da pista. A velocidade média de 0.34 mostra que a política ficou mais conservadora nessa pista, provavelmente por causa da dificuldade maior e das combinações de curvas e estreitamentos.

A recompensa total ficou negativa, em `-29.00`, o que indica que o agente não conseguiu transformar o progresso em uma trajetória eficiente até a chegada. A pista 18 parece exigir uma coordenação melhor entre acelerar nas retas e reduzir velocidade nos trechos mais apertados.

## 4.3 Análise crítica

O resultado mostra uma generalização parcial. A pista 17 não foi usada no treinamento, e mesmo assim o agente conseguiu chegar ao final. Isso indica que a representação por sensores LIDAR funcionou para transferir alguns comportamentos aprendidos, como reagir a paredes próximas, virar em curvas e controlar a velocidade em certos momentos.

Por outro lado, a falha na pista 18 mostra uma limitação importante da abordagem. O agente não conhece o mapa completo, não sabe sua posição absoluta e também não possui memória. Ele decide apenas com base nas leituras locais dos sensores e na velocidade atual. Em pistas mais longas ou com padrões mais complexos, essa informação local pode não ser suficiente para uma política tabular funcionar bem.

Também é importante notar que o Q-Learning tabular não generaliza entre estados de forma suave como uma rede neural faria. Dois estados parecidos, mas que caem em baldes diferentes, são tratados como entradas separadas na tabela. Isso ajuda a explicar por que o agente consegue aprender alguns padrões, mas ainda falha quando a pista exige combinações mais específicas de ações.

A pista 18 provavelmente exigiria mais treinamento, ajustes na recompensa ou alguma melhoria na estratégia de exploração. Também seria possível testar mudanças como aumentar o número de episódios, ajustar o decaimento do epsilon, avaliar mais episódios por pista ou experimentar outra discretização. Mesmo assim, o resultado obtido já demonstra aprendizado real e uma capacidade parcial de generalização.

---

## 5. Arquivos gerados

A solução gera os seguintes arquivos principais:

```text
treinamento/qlearning.pkl
q_learning_pista_17.txt
q_learning_pista_18.txt
```

O arquivo `qlearning.pkl` contém o modelo treinado. Nele ficam a tabela Q, a discretização usada, os hiperparâmetros, a seed, o histórico de recompensas, o histórico de sucessos e a lista de pistas usadas no treinamento.

Os arquivos `q_learning_pista_17.txt` e `q_learning_pista_18.txt` registram as métricas finais da avaliação nas pistas de holdout.

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

Para treinar novamente com a mesma configuração usada neste relatório:

```bash
python solucao.py --episodios-por-pista 5000 --recarregar
```

Para carregar o modelo salvo e avaliar nas pistas 17 e 18:

```bash
python solucao.py
```

Para avaliar uma pista específica usando o modelo salvo:

```bash
python solucao.py --avaliar pistas/pista_17.txt
```

---

## 7. Conclusão

A implementação conseguiu cumprir a proposta principal do EP: construir um agente de Q-Learning tabular do zero, discretizar o estado contínuo, treinar em múltiplas pistas com round-robin, salvar o modelo em pickle e avaliar a política aprendida em pistas de holdout.

O principal resultado foi o sucesso na pista 17, que não fez parte do treinamento. Isso mostra que o agente aprendeu padrões locais úteis a partir dos sensores LIDAR e conseguiu aplicar esses padrões em uma pista nova.

Na pista 18, o agente não chegou ao final dentro do limite de passos. Essa falha não invalida o aprendizado, mas mostra os limites da solução tabular com estado local. A política aprendida foi suficiente para generalizar em uma pista holdout, mas ainda não foi robusta o bastante para lidar com a pista mais difícil.

No geral, o trabalho mostra que o Q-Learning tabular pode funcionar para esse tipo de ambiente, desde que o estado seja bem discretizado e o treinamento tenha episódios suficientes. Ao mesmo tempo, os resultados deixam claro que a escolha da representação do estado, do orçamento de treino e da exploração influencia bastante o desempenho final.
