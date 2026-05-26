"""
Solução para o EP do carrinho autônomo usando Q-Learning tabular.

Fluxo principal:
1. Treina um agente Q-Learning nas pistas 01-16.
2. Salva a tabela Q em treinamento/qlearning.pkl.
3. Avalia o agente, com política gulosa, nas pistas holdout 17 e 18.
4. Gera q_learning_pista_17.txt e q_learning_pista_18.txt.
"""

import sys
import random
import argparse
import pickle
from pathlib import Path

import numpy as np

# Adiciona src ao path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from env import AmbienteCarro  # noqa: E402


# ============================================================================
# CONFIGURAÇÕES GERAIS
# ============================================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DIR_TREINAMENTO = Path("treinamento")
DIR_TREINAMENTO.mkdir(exist_ok=True)

PISTAS_TREINO = [f"pistas/pista_{i:02d}.txt" for i in range(1, 17)]
PISTAS_HOLDOUT = [f"pistas/pista_{i:02d}.txt" for i in range(17, 19)]


# ============================================================================
# AGENTE Q-LEARNING TABULAR
# ============================================================================

class AgenteQLearning:
    """
    Agente Q-Learning tabular.

    O ambiente retorna um vetor contínuo:
        [d_0, d_+30, d_-30, d_+60, d_-60, v_norm]

    Como Q-Learning tabular precisa de estados discretos,
    cada valor é colocado em um dos K baldes.
    """

    def __init__(
        self,
        obs_dim,
        n_actions,
        K=5,
        alpha=0.2,
        gamma=0.95,
        eps_inicial=1.0,
        eps_final=0.15,
    ):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.K = K
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps_inicial
        self.eps_inicial = eps_inicial
        self.eps_final = eps_final
        self.Q = {}

    def discretizar(self, obs):
        """
        Converte observação contínua em chave discreta.
        """
        chave = []

        for v in obs:
            v = float(v)
            v = max(0.0, min(1.0, v))
            balde = min(int(v * self.K), self.K - 1)
            chave.append(balde)

        return tuple(chave)

    def valores_iniciais_q(self):
        """
        Q-values iniciais para estados novos.

        Pequeno viés em acelerar e virar evita que o agente fique parado
        quando ainda não conhece um estado.
        """
        return np.array([0.0, 0.20, 0.0, 0.05, 0.05], dtype=np.float32)

    def obter_q(self, chave):
        """
        Retorna os Q-values do estado e cria a chave se ela ainda não existir.
        Usado durante o treinamento.
        """
        if chave not in self.Q:
            self.Q[chave] = self.valores_iniciais_q()

        return self.Q[chave]

    def consultar_q(self, chave):
        """
        Retorna os Q-values do estado sem alterar a tabela Q.
        Usado durante a avaliação.
        """
        return self.Q.get(chave, self.valores_iniciais_q())

    def escolher_acao(self, obs, criar_estado=True):
        """
        Política epsilon-greedy.

        criar_estado=True no treino.
        criar_estado=False na avaliação, para não alterar a Q-table.
        """
        chave = self.discretizar(obs)

        if criar_estado:
            q_vals = self.obter_q(chave)
        else:
            q_vals = self.consultar_q(chave)

        if random.random() < self.eps:
            return random.randint(0, self.n_actions - 1)

        melhor_valor = np.max(q_vals)
        melhores_acoes = np.flatnonzero(q_vals == melhor_valor)

        return int(np.random.choice(melhores_acoes))

    def atualizar(self, s, a, r, s_prox, terminou):
        """
        Atualização do Q-Learning:

        Q(s,a) <- Q(s,a) + alpha * [r + gamma * max Q(s',a') - Q(s,a)]
        """
        chave = self.discretizar(s)
        chave_prox = self.discretizar(s_prox)

        q_atual = self.obter_q(chave)
        q_prox = self.obter_q(chave_prox)

        if terminou:
            alvo = r
        else:
            alvo = r + self.gamma * np.max(q_prox)

        erro_td = alvo - q_atual[a]
        q_atual[a] += self.alpha * erro_td

    @classmethod
    def from_modelo(cls, modelo):
        """
        Reconstrói um agente a partir do pickle salvo.
        """
        config = modelo.get("config", {})

        agente = cls(
            obs_dim=6,
            n_actions=5,
            K=modelo.get("discretization_K", 5),
            alpha=config.get("alpha", 0.2),
            gamma=config.get("gamma", 0.95),
            eps_inicial=0.0,
            eps_final=0.0,
        )

        agente.Q = modelo["q_table"]
        agente.eps = 0.0

        return agente


# ============================================================================
# TREINAMENTO
# ============================================================================

def atualizar_epsilon(agente, episodio_atual, episodios_decaimento):
    """
    Decaimento linear do epsilon.
    """
    if episodios_decaimento <= 0:
        agente.eps = agente.eps_final
        return

    progresso = min(episodio_atual / episodios_decaimento, 1.0)

    agente.eps = agente.eps_inicial - (
        agente.eps_inicial - agente.eps_final
    ) * progresso

    agente.eps = max(agente.eps, agente.eps_final)


def rodar_episodio_treino(env, agente):
    """
    Executa um episódio de treino e atualiza a tabela Q.
    """
    obs = env.reset()
    done = False
    recompensa_total = 0.0
    sucesso = False

    while not done:
        action = agente.escolher_acao(obs, criar_estado=True)
        obs_prox, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Truncamento também é tratado como fim do episódio para evitar
        # superestimar estados nos quais o agente ficou travado até max_steps.
        agente.atualizar(obs, action, reward, obs_prox, done)

        obs = obs_prox
        recompensa_total += reward

        if info.get("chegada"):
            sucesso = True

    return recompensa_total, sucesso


def treinar_round_robin(
    pistas_treino,
    agente,
    n_episodios_por_pista,
    max_passos,
    decaimento_eps_episodios,
    verbose=True,
):
    """
    Treinamento round-robin.
    A cada episódio, sorteia uma pista entre as pistas de treino.
    """
    historico_recompensas = []
    historico_sucessos = []
    rewards_por_pista = {p: [] for p in pistas_treino}
    sucessos_por_pista = {p: [] for p in pistas_treino}

    n_total = n_episodios_por_pista * len(pistas_treino)

    envs = {
        p: AmbienteCarro(p, max_steps=max_passos, seed=SEED)
        for p in pistas_treino
    }

    for ep in range(n_total):
        atualizar_epsilon(agente, ep, decaimento_eps_episodios)

        pista = random.choice(pistas_treino)
        env = envs[pista]

        recompensa_total, sucesso = rodar_episodio_treino(env, agente)

        historico_recompensas.append(recompensa_total)
        historico_sucessos.append(sucesso)
        rewards_por_pista[pista].append(recompensa_total)
        sucessos_por_pista[pista].append(sucesso)

        if verbose and (ep + 1) % 1000 == 0:
            janela_rewards = historico_recompensas[-100:]
            janela_sucessos = historico_sucessos[-100:]

            media_reward = sum(janela_rewards) / len(janela_rewards)
            taxa_sucesso = sum(janela_sucessos) / len(janela_sucessos)

            print(
                f"Episódio {ep + 1}/{n_total} | "
                f"eps={agente.eps:.3f} | "
                f"reward médio 100={media_reward:.2f} | "
                f"sucesso 100={taxa_sucesso:.2%} | "
                f"estados={len(agente.Q)}"
            )

    return historico_recompensas, historico_sucessos, rewards_por_pista, sucessos_por_pista


# ============================================================================
# AVALIAÇÃO
# ============================================================================

def avaliar(env, agente, n_episodios=10):
    """
    Avalia o agente com epsilon = 0.
    """
    eps_antigo = agente.eps
    agente.eps = 0.0

    resultados = []

    for _ in range(n_episodios):
        obs = env.reset()
        done = False
        recompensa_total = 0.0
        velocidades = []
        velocidade_maxima = 0.0
        sucesso = False

        while not done:
            action = agente.escolher_acao(obs, criar_estado=False)
            obs, reward, terminated, truncated, info = env.step(action)

            recompensa_total += reward

            if env.carro is not None:
                velocidades.append(env.carro.v)
                velocidade_maxima = max(velocidade_maxima, env.carro.v)

            if info.get("chegada"):
                sucesso = True

            done = terminated or truncated

        velocidade_media = (
            sum(velocidades) / len(velocidades)
            if velocidades
            else 0.0
        )

        resultados.append({
            "n_passos": env.passos,
            "recompensa_total": recompensa_total,
            "sucesso": sucesso,
            "velocidade_media": velocidade_media,
            "velocidade_maxima": velocidade_maxima,
        })

    agente.eps = eps_antigo

    sucessos = [r for r in resultados if r["sucesso"]]

    if sucessos:
        return min(sucessos, key=lambda r: r["n_passos"])

    return max(resultados, key=lambda r: r["recompensa_total"])


def avaliar_varias_pistas(pistas, agente, max_passos):
    """
    Avalia o agente em várias pistas e imprime um resumo no terminal.
    """
    print("\nResumo de avaliação:")
    print("-" * 72)
    print(f"{'Pista':<16} {'Sucesso':<10} {'Passos':<10} {'Reward':<12} {'Vel. média':<12}")
    print("-" * 72)

    resultados = {}

    for pista in pistas:
        env = AmbienteCarro(pista, max_steps=max_passos, seed=SEED)
        resultado = avaliar(env, agente)
        resultado["estados_populados"] = len(agente.Q)
        resultados[pista] = resultado

        sucesso_txt = "SIM" if resultado["sucesso"] else "NAO"

        print(
            f"{Path(pista).name:<16} "
            f"{sucesso_txt:<10} "
            f"{resultado['n_passos']:<10} "
            f"{resultado['recompensa_total']:<12.2f} "
            f"{resultado['velocidade_media']:<12.2f}"
        )

    print("-" * 72)

    return resultados


# ============================================================================
# SALVAR / CARREGAR
# ============================================================================

def treinar_ou_carregar(nome, fn_treinar, recarregar=False):
    """
    Carrega treinamento/{nome}.pkl se existir.
    Se recarregar=True ou o arquivo não existir, treina novamente.
    """
    arquivo = DIR_TREINAMENTO / f"{nome}.pkl"

    if arquivo.exists() and not recarregar:
        print(f"Carregando {arquivo} ...")

        with open(arquivo, "rb") as f:
            return pickle.load(f)

    print(f"Treinando {nome} ...")
    resultado = fn_treinar()

    with open(arquivo, "wb") as f:
        pickle.dump(resultado, f)

    print(f"Salvo em {arquivo}")

    return resultado


# ============================================================================
# ARQUIVOS DE SAÍDA
# ============================================================================

def escrever_saida(
    caminho,
    nome_algoritmo,
    pista,
    resultado_avaliacao,
    n_episodios_treinados,
):
    """
    Escreve arquivo de saída no formato pedido no enunciado.
    """
    sucesso_txt = "SIM" if resultado_avaliacao["sucesso"] else "NAO"
    estados_populados = resultado_avaliacao.get("estados_populados", "N/A")

    conteudo = (
        f"=== Pista: {Path(pista).name} ===\n"
        f"Algoritmo: {nome_algoritmo} (round-robin em pistas 01-16)\n"
        f"Episódios totais de treinamento: {n_episodios_treinados}\n"
        f"Estados populados: {estados_populados}\n"
        f"Tempo de chegada (passos): {resultado_avaliacao['n_passos']}\n"
        f"Velocidade média: {resultado_avaliacao['velocidade_media']:.2f}\n"
        f"Velocidade máxima atingida: {resultado_avaliacao['velocidade_maxima']:.2f}\n"
        f"Recompensa total: {resultado_avaliacao['recompensa_total']:.2f}\n"
        f"Sucesso: {sucesso_txt}\n"
    )

    with open(caminho, "w", encoding="utf-8") as f:
        f.write(conteudo)

    print(f"Arquivo gerado: {caminho}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--episodios-por-pista",
        type=int,
        default=30_000,
        help="Episódios por pista no treino round-robin."
    )
    parser.add_argument(
        "--max-passos",
        type=int,
        default=500,
        help="Limite de passos por episódio."
    )
    parser.add_argument(
        "--K",
        type=int,
        default=5,
        help="Quantidade de baldes da discretização."
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.2,
        help="Taxa de aprendizado."
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.95,
        help="Fator de desconto."
    )
    parser.add_argument(
        "--eps-inicial",
        type=float,
        default=1.0,
        help="Epsilon inicial da política epsilon-greedy."
    )
    parser.add_argument(
        "--eps-final",
        type=float,
        default=0.15,
        help="Epsilon final da política epsilon-greedy."
    )
    parser.add_argument(
        "--recarregar",
        action="store_true",
        help="Força re-treino mesmo se já existir pickle."
    )
    parser.add_argument(
        "--avaliar",
        type=str,
        default=None,
        help="Avalia apenas uma pista específica usando o modelo salvo."
    )
    parser.add_argument(
        "--avaliar-treino",
        action="store_true",
        help="Depois do treino, também avalia todas as pistas de treino."
    )

    args = parser.parse_args()

    def fn_treinar():
        agente = AgenteQLearning(
            obs_dim=6,
            n_actions=5,
            K=args.K,
            alpha=args.alpha,
            gamma=args.gamma,
            eps_inicial=args.eps_inicial,
            eps_final=args.eps_final,
        )

        n_total = args.episodios_por_pista * len(PISTAS_TREINO)

        rewards, sucessos, rewards_por_pista, sucessos_por_pista = treinar_round_robin(
            pistas_treino=PISTAS_TREINO,
            agente=agente,
            n_episodios_por_pista=args.episodios_por_pista,
            max_passos=args.max_passos,
            decaimento_eps_episodios=int(0.8 * n_total),
        )

        modelo = {
            "q_table": agente.Q,
            "discretization_K": args.K,
            "n_episodes_trained": n_total,
            "rewards_history": rewards,
            "success_history": sucessos,
            "rewards_por_pista": rewards_por_pista,
            "sucessos_por_pista": sucessos_por_pista,
            "config": {
                "alpha": agente.alpha,
                "gamma": agente.gamma,
                "eps_inicial": agente.eps_inicial,
                "eps_final": agente.eps_final,
                "max_passos": args.max_passos,
                "K": args.K,
            },
            "seed": SEED,
            "tracks_used": PISTAS_TREINO,
            "n_estados_populados": len(agente.Q),
        }

        return modelo

    modelo = treinar_ou_carregar(
        "qlearning",
        fn_treinar,
        recarregar=args.recarregar
    )

    agente_avaliacao = AgenteQLearning.from_modelo(modelo)

    if args.avaliar_treino:
        avaliar_varias_pistas(
            PISTAS_TREINO,
            agente_avaliacao,
            args.max_passos
        )

    pistas_avaliar = [args.avaliar] if args.avaliar else PISTAS_HOLDOUT

    for pista in pistas_avaliar:
        env = AmbienteCarro(pista, max_steps=args.max_passos, seed=SEED)
        resultado = avaliar(env, agente_avaliacao)
        resultado["estados_populados"] = len(agente_avaliacao.Q)

        nome_pista = Path(pista).stem

        escrever_saida(
            caminho=f"q_learning_{nome_pista}.txt",
            nome_algoritmo="Q-Learning",
            pista=pista,
            resultado_avaliacao=resultado,
            n_episodios_treinados=modelo["n_episodes_trained"],
        )

    print("\nPronto.")


if __name__ == "__main__":
    main()
