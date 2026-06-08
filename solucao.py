"""
Esqueleto da sua solução para o EP do carrinho (versão tabular).

Você deve implementar:
    - AgenteQLearning  (tabular)

E preencher main() para orquestrar:
    1. Treinamento round-robin nas pistas 01-16 → salva treinamento/qlearning.pkl.
    2. Avaliação gulosa (ε = 0) nas pistas de holdout 17 e 18 → gera
       q_learning_pista_17.txt e q_learning_pista_18.txt (formato do README §4.3).

Uso:
    python solucao.py                         # treina (se necessário) + avalia em 17 e 18
    python solucao.py --recarregar            # força re-treino (ignora pickle existente)
    python solucao.py --avaliar pistas/X.txt  # apenas avalia o modelo salvo em X

Termos como `step`, `reset`, `obs`, `action`, `reward` são mantidos em inglês
por serem o vocabulário canônico de Aprendizado por Reforço (Sutton & Barto).
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
# from visualize import renderizar_episodio  # use isto para animar seu agente no terminal


# === Configuração ===
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Diretório onde o modelo treinado será salvo via pickle (ver enunciado/anexo_b_pickle.md)
DIR_TREINAMENTO = Path("treinamento")
DIR_TREINAMENTO.mkdir(exist_ok=True)

# Conjuntos de pistas
PISTAS_TREINO = [f"pistas/pista_{i:02d}.txt" for i in range(1, 17)]   # 01..16
PISTAS_HOLDOUT = [f"pistas/pista_{i:02d}.txt" for i in range(17, 19)] # 17, 18


# ============================================================================
# Q-LEARNING TABULAR
# ============================================================================

class AgenteQLearning:
    """
    Agente Q-Learning tabular com discretização uniforme do estado.
    """

    def __init__(self, obs_dim, n_actions, K=5, alpha=0.1, gamma=0.99,
                 eps_inicial=1.0, eps_final=0.05):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.K = K
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps_inicial
        self.eps_inicial = eps_inicial
        self.eps_final = eps_final

        # Tabela Q:
        # chave: estado discretizado, exemplo: (4, 2, 1, 1, 0, 0)
        # valor: array com um valor para cada ação
        self.Q = {}

    def discretizar(self, obs):
        """
        Converte o vetor de 6 floats em uma tupla de inteiros.
        Cada valor em [0, 1] vira um balde entre 0 e K-1.
        """
        return tuple(
            min(int(float(v) * self.K), self.K - 1)
            for v in obs
        )

    def obter_q(self, chave):
        """
        Se o estado ainda não existe na tabela Q, cria com valores zerados.
        """
        if chave not in self.Q:
            self.Q[chave] = np.zeros(self.n_actions, dtype=np.float32)
        return self.Q[chave]

    def escolher_acao(self, obs):
        """
        Política epsilon-greedy.
        Com chance eps, explora.
        Caso contrário, escolhe a melhor ação conhecida.
        """
        chave = self.discretizar(obs)
        q_vals = self.obter_q(chave)

        if random.random() < self.eps:
            return random.randint(0, self.n_actions - 1)

        return int(np.argmax(q_vals))

    def atualizar(self, s, a, r, s_prox, terminou):
        """
        Atualização do Q-Learning:

        Q(s,a) = Q(s,a) + alpha * [r + gamma * max Q(s',a') - Q(s,a)]
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
        Reconstrói o agente a partir do pickle salvo.
        """
        config = modelo.get("config", {})

        agente = cls(
            obs_dim=6,
            n_actions=5,
            K=modelo.get("discretization_K", 5),
            alpha=config.get("alpha", 0.1),
            gamma=config.get("gamma", 0.99),
            eps_inicial=0.0,
            eps_final=0.0,
        )

        agente.Q = modelo["q_table"]
        agente.eps = 0.0

        return agente


# ============================================================================
# LOOP DE TREINAMENTO (round-robin nas 16 pistas de treino)
# ============================================================================

def treinar_round_robin(pistas_treino, agente, n_episodios_por_pista,
                       max_passos, decaimento_eps_episodios, verbose=True):
    """
    Treina o agente sorteando uma pista de treino a cada episódio.
    Isso evita que ele aprenda uma pista e esqueça as anteriores.
    """
    historico_recompensas = []
    historico_sucessos = []
    rewards_por_pista = {p: [] for p in pistas_treino}

    n_total = n_episodios_por_pista * len(pistas_treino)

    envs = {
        p: AmbienteCarro(p, max_steps=max_passos, seed=SEED)
        for p in pistas_treino
    }

    for ep in range(n_total):
        # Decaimento linear do epsilon
        progresso_decay = min(ep / decaimento_eps_episodios, 1.0)

        agente.eps = agente.eps_inicial - (
            agente.eps_inicial - agente.eps_final
        ) * progresso_decay

        agente.eps = max(agente.eps, agente.eps_final)

        # Sorteia uma pista de treino
        pista = random.choice(pistas_treino)
        env = envs[pista]

        obs = env.reset()
        done = False
        recompensa_total = 0.0
        sucesso = False

        while not done:
            action = agente.escolher_acao(obs)

            obs_prox, reward, terminated, truncated, info = env.step(action)

            # Para Q-Learning, terminal real é colisão ou chegada.
            # Truncamento é só limite de passos.
            agente.atualizar(obs, action, reward, obs_prox, terminated)

            obs = obs_prox
            recompensa_total += reward

            done = terminated or truncated

            if info.get("chegada"):
                sucesso = True

        historico_recompensas.append(recompensa_total)
        historico_sucessos.append(sucesso)
        rewards_por_pista[pista].append(recompensa_total)

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

    return historico_recompensas, historico_sucessos, rewards_por_pista


# ============================================================================
# AVALIAÇÃO (com ε = 0)
# ============================================================================

def avaliar(env, agente, n_episodios=10):
    """
    Roda n_episodios com política gulosa, epsilon = 0,
    e retorna estatísticas da melhor tentativa.
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
            action = agente.escolher_acao(obs)
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
            if velocidades else 0.0
        )

        resultados.append({
            "n_passos": env.passos,
            "recompensa_total": recompensa_total,
            "sucesso": sucesso,
            "velocidade_media": velocidade_media,
            "velocidade_maxima": velocidade_maxima,
        })

    agente.eps = eps_antigo

    # Prioriza episódios com sucesso.
    # Se nenhum teve sucesso, pega o de maior recompensa.
    sucessos = [r for r in resultados if r["sucesso"]]

    if sucessos:
        return min(sucessos, key=lambda r: r["n_passos"])

    return max(resultados, key=lambda r: r["recompensa_total"])


# ============================================================================
# SALVAR / CARREGAR MODELO (ver enunciado/anexo_b_pickle.md)
# ============================================================================

def treinar_ou_carregar(nome, fn_treinar, recarregar=False):
    """
    Se 'treinamento/{nome}.pkl' existe e recarregar=False, carrega.
    Caso contrário, chama fn_treinar() e salva o resultado.
    """
    arquivo = DIR_TREINAMENTO / f"{nome}.pkl"
    if arquivo.exists() and not recarregar:
        print(f"Carregando {arquivo} ...")
        with open(arquivo, "rb") as f:
            return pickle.load(f)
    else:
        print(f"Treinando {nome} ...")
        resultado = fn_treinar()
        with open(arquivo, "wb") as f:
            pickle.dump(resultado, f)
        print(f"Salvo em {arquivo}")
        return resultado


# ============================================================================
# GERAÇÃO DOS ARQUIVOS DE SAÍDA
# ============================================================================

def escrever_saida(caminho, nome_algoritmo, pista, resultado_avaliacao, n_episodios_treinados):
    """
    Escreve um arquivo no formato esperado pelo README.
    """
    sucesso_txt = "SIM" if resultado_avaliacao["sucesso"] else "NAO"

    estados_populados = resultado_avaliacao.get("estados_populados", None)

    conteudo = (
        f"=== Pista: {Path(pista).name} ===\n"
        f"Algoritmo: {nome_algoritmo} (round-robin em pistas 01-16)\n"
        f"Episódios totais de treinamento: {n_episodios_treinados}\n"
        f"Estados populados: {resultado_avaliacao.get('estados_populados', 'N/A')}\n"
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
    parser.add_argument("--episodios-por-pista", type=int, default=30_000,
                        help="Episódios de treino por pista no round-robin")
    parser.add_argument("--max-passos", type=int, default=500)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--recarregar", action="store_true",
                        help="Força re-treino mesmo se o pickle existir")
    parser.add_argument("--avaliar", type=str, default=None,
                        help="Apenas avalia o modelo salvo na pista especificada")
    args = parser.parse_args()

    def fn_treinar():
        agente = AgenteQLearning(
            obs_dim=6,
            n_actions=5,
            K=args.K,
            alpha=0.1,
            gamma=0.99,
            eps_inicial=1.0,
            eps_final=0.05,
        )

        n_total = args.episodios_por_pista * len(PISTAS_TREINO)

        rewards, sucessos, rewards_por_pista = treinar_round_robin(
            PISTAS_TREINO,
            agente,
            args.episodios_por_pista,
            args.max_passos,
            decaimento_eps_episodios=int(0.8 * n_total),
        )

        return {
            "q_table": agente.Q,
            "discretization_K": args.K,
            "n_episodes_trained": n_total,
            "rewards_history": rewards,
            "success_history": sucessos,
            "rewards_por_pista": rewards_por_pista,
            "config": {
                "alpha": agente.alpha,
                "gamma": agente.gamma,
                "eps_inicial": agente.eps_inicial,
                "eps_final": agente.eps_final,
                "max_passos": args.max_passos,
            },
            "seed": SEED,
            "tracks_used": PISTAS_TREINO,
        }

    modelo = treinar_ou_carregar(
        "qlearning",
        fn_treinar,
        recarregar=args.recarregar
    )

    agente_avaliacao = AgenteQLearning.from_modelo(modelo)

    pistas_avaliar = [args.avaliar] if args.avaliar else PISTAS_HOLDOUT

    for pista in pistas_avaliar:
        env = AmbienteCarro(pista, max_steps=args.max_passos, seed=SEED)
        resultado = avaliar(env, agente_avaliacao)
        resultado["estados_populados"] = len(modelo["q_table"])

        nome_pista = Path(pista).stem

        escrever_saida(
            f"q_learning_{nome_pista}.txt",
            "Q-Learning",
            pista,
            resultado,
            modelo["n_episodes_trained"]
        )

    print("\nPronto.")

if __name__ == "__main__":
    main()
