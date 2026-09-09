"""
desafio1_aluno.py — Desafio 1: Mapa de características para o Perceptron
GBC073 — Inteligência Computacional (FACOM/UFU)

O QUE VOCÊ FAZ: preencher a classe `Submissao` (e só ela).
O QUE O HARNESS FAZ: gera dados, divide 60/40, aplica a sua phi, treina um
Perceptron fixo e mede a acurácia. Rode:  python desafio1_aluno.py

Regras:
  * fit(X) recebe só as entradas de treino, sem rótulos. É opcional.
  * phi(X) transforma (n, d) em (n, d') com d < d' <= 64, de forma determinística.
  * Sem NaN/Inf; e rápido (10 mil pontos em menos de 2 s).
Escore: 0 = igual à identidade (baseline), 100 = igual à referência do professor,
até 125 se superar a referência. Na correção, tarefas OCULTAS da mesma família
substituem estas — não ajuste para um conjunto de dados específico.
"""
import math
import time
import torch

DIM_MAX = 64
SEMENTES = (0, 1, 2)

# =============================================================================
# >>> SUA SUBMISSÃO — edite apenas esta classe <<<
# =============================================================================
class Submissao:
    DIM_MAX = DIM_MAX

    def fit(self, X: torch.Tensor) -> None:
        d = X.shape[1]
        self.d = d
        self.mu = X.mean(0)
        self.sd = X.std(0) + 1e-8
        Xs = (X - self.mu) / self.sd
        g = torch.Generator().manual_seed(42)

        # Warmup de operações matriciais
        _ = Xs[:2] @ Xs[:2].T

        # Norma radial ao quadrado (essencial para hiperesferas, círculos e densidade)
        r2 = (Xs ** 2).sum(dim=1, keepdim=True)
        self.r2_mu = r2.mean()
        self.r2_sd = r2.std() + 1e-8

        if d == 2:
            prod = Xs[:, 0:1] * Xs[:, 1:2]
            self.prod_mu = prod.mean()
            self.prod_sd = prod.std() + 1e-8

            th = torch.atan2(Xs[:, 1:2], Xs[:, 0:1]) / math.pi
            self.th_mu = th.mean()
            self.th_sd = th.std() + 1e-8

            # Base 2D: Xs (2), r2 (1), prod (1), theta (1) = 5
            n_base = 5
        else:
            sq = Xs ** 2
            self.sq_mu = sq.mean(0)
            self.sq_sd = sq.std(0) + 1e-8
            # Base d > 2: Xs (d), r2 (1), sq (d) = 2d + 1
            n_base = 2 * d + 1

        m_landmarks = self.DIM_MAX - n_base

        # Seleção de marcos representativos (k-means++ sobre os dados não rotulados)
        n = len(Xs)
        idx0 = torch.randint(0, n, (1,), generator=g).item()
        centers = [Xs[idx0]]
        min_dists = torch.cdist(Xs, Xs[idx0:idx0+1]).squeeze(1) ** 2
        for _ in range(1, m_landmarks):
            probs = min_dists / (min_dists.sum() + 1e-8)
            next_idx = torch.multinomial(probs, 1, generator=g).item()
            centers.append(Xs[next_idx])
            dists_new = torch.cdist(Xs, Xs[next_idx:next_idx+1]).squeeze(1) ** 2
            min_dists = torch.minimum(min_dists, dists_new)

        self.centers = torch.stack(centers)

        # Largura de banda multiescala (quantis de distância para curvas locais e globais)
        idx_sub = torch.randperm(n, generator=g)[:200]
        p_dist = torch.cdist(Xs[idx_sub], Xs[idx_sub])
        p_dist_pos = p_dist[p_dist > 0]

        escalas = [0.05, 0.15, 0.35, 0.70]
        sigmas = [torch.quantile(p_dist_pos, q).item() for q in escalas]
        gammas = []
        rest = m_landmarks
        for k, sigma in enumerate(sigmas):
            mk = rest // (len(sigmas) - k)
            rest -= mk
            sigma = max(sigma, 1e-4)
            gammas.extend([1.0 / (2 * (sigma ** 2))] * mk)

        self.gamma = torch.tensor(gammas).unsqueeze(0)

        # Estatísticas de padronização do RBF para convergência equilibrada do Perceptron
        rbf_tr = torch.exp(-self.gamma * (torch.cdist(Xs, self.centers) ** 2))
        self.rbf_mu = rbf_tr.mean(0, keepdim=True)
        self.rbf_sd = rbf_tr.std(0, keepdim=True) + 1e-8

    def phi(self, X: torch.Tensor) -> torch.Tensor:
        Xs = (X - self.mu) / self.sd
        r2 = (Xs ** 2).sum(dim=1, keepdim=True)
        r2_norm = (r2 - self.r2_mu) / self.r2_sd

        if self.d == 2:
            prod = Xs[:, 0:1] * Xs[:, 1:2]
            prod_norm = (prod - self.prod_mu) / self.prod_sd
            th = torch.atan2(Xs[:, 1:2], Xs[:, 0:1]) / math.pi
            th_norm = (th - self.th_mu) / self.th_sd
            base = [Xs, 2.0 * r2_norm, 2.0 * prod_norm, th_norm]
        else:
            sq = Xs ** 2
            sq_norm = (sq - self.sq_mu) / self.sq_sd
            base = [Xs, 2.0 * r2_norm, sq_norm]

        # Núcleo de Base Radial (RBF) em relação aos centros de referência
        dist2 = torch.cdist(Xs, self.centers) ** 2
        rbf = torch.exp(-self.gamma * dist2)
        rbf_norm = (rbf - self.rbf_mu) / self.rbf_sd

        return torch.cat(base + [0.8 * rbf_norm], dim=1)



# =============================================================================
# Harness (não edite daqui para baixo)
# =============================================================================
def _luas(n, g):
    t = torch.rand(n // 2, generator=g) * math.pi
    X = torch.cat([torch.stack([torch.cos(t), torch.sin(t)], 1),
                   torch.stack([1 - torch.cos(t), 0.5 - torch.sin(t)], 1)])
    y = torch.cat([torch.zeros(n // 2), torch.ones(n // 2)])
    return X + 0.15 * torch.randn(n, 2, generator=g), y

def _circulos(n, g):
    t = torch.rand(n, generator=g) * 2 * math.pi
    r = torch.where(torch.arange(n) < n // 2, 1.0, 0.45)
    X = torch.stack([r * torch.cos(t), r * torch.sin(t)], 1)
    return X + 0.08 * torch.randn(n, 2, generator=g), (torch.arange(n) >= n // 2).float()

def _xor(n, g):
    X = torch.rand(n, 2, generator=g) * 2 - 1
    y = (X[:, 0] * X[:, 1] < 0).float()
    return X + 0.15 * torch.randn(n, 2, generator=g), y

def _espiral(n, g):
    t = torch.sqrt(torch.rand(n // 2, generator=g)) * 3 * math.pi
    a = torch.stack([t * torch.cos(t), t * torch.sin(t)], 1) / 10
    y = torch.cat([torch.zeros(n // 2), torch.ones(n // 2)])
    return torch.cat([a, -a]) + 0.05 * torch.randn(n, 2, generator=g), y

def _esfera(n, g, d=10):
    X = torch.randn(n, d, generator=g)
    r2 = (X ** 2).sum(1)
    return X, (r2 > r2.median()).float()

TAREFAS = {"xor": lambda g: _xor(600, g), "duas_luas": lambda g: _luas(600, g),
           "circulos": lambda g: _circulos(600, g), "espiral": lambda g: _espiral(800, g),
           "esfera_10d": lambda g: _esfera(800, g)}


@torch.no_grad()
def perceptron_pocket(Z, y, epocas=50, eta=1.0, semente=0):
    """Regra de Rosenblatt (w <- w + eta*y*z nos erros) + pocket: guarda o melhor w."""
    Zb = torch.cat([Z, torch.ones(len(Z), 1)], 1)     # viés embutido
    yb = 2 * y - 1
    w = torch.zeros(Zb.shape[1]); melhor_w, melhor_acc = w.clone(), -1.0
    g = torch.Generator().manual_seed(semente)
    for _ in range(epocas):
        for i in torch.randperm(len(Zb), generator=g).tolist():
            if yb[i] * (Zb[i] @ w) <= 0:
                w += eta * yb[i] * Zb[i]
        acc = ((Zb @ w) * yb > 0).float().mean().item()
        if acc > melhor_acc:
            melhor_acc, melhor_w = acc, w.clone()
    return melhor_w


def acuracia_balanceada(y, yhat):
    return torch.stack([(yhat[y == c] == c).float().mean() for c in y.unique()]).mean().item()


@torch.no_grad()
def rodar(sub, gerador, semente, checar=True):
    g = torch.Generator().manual_seed(semente)
    X, y = gerador(g)
    idx = torch.randperm(len(X), generator=g); ntr = int(0.6 * len(X))
    Xtr, ytr, Xte, yte = X[idx[:ntr]], y[idx[:ntr]], X[idx[ntr:]], y[idx[ntr:]]

    torch.manual_seed(semente)
    sub.fit(Xtr)                                       # nunca recebe ytr
    t0 = time.perf_counter(); Ztr = sub.phi(Xtr); dt = time.perf_counter() - t0
    Zte = sub.phi(Xte)
    if checar:                                         # regras do desafio
        d, dl = Xtr.shape[1], Ztr.shape[1]
        assert Ztr.ndim == 2 and Zte.shape[1] == dl, "phi deve devolver (n, d')"
        assert d < dl <= DIM_MAX, f"exige d < d' <= {DIM_MAX}; recebi d={d}, d'={dl}"
        assert torch.isfinite(Ztr).all() and torch.isfinite(Zte).all(), "NaN/Inf na saída de phi"
        assert torch.allclose(sub.phi(Xtr[:20]), Ztr[:20]), "phi não é determinística"
        assert dt * (10_000 / len(Xtr)) < 2.0, "phi lenta demais (limite: 10^4 pontos em 2 s)"

    w = perceptron_pocket(Ztr, ytr, semente=semente)
    yhat = (torch.cat([Zte, torch.ones(len(Zte), 1)], 1) @ w > 0).float()
    return acuracia_balanceada(yte, yhat)


class _Identidade:                       # baseline (viola d < d', mas é só o ponto zero da escala)
    def fit(self, X): pass
    def phi(self, X): return X

class _RFF:                              # referência: random Fourier features, d' = 64
    def fit(self, X):
        g = torch.Generator().manual_seed(0)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-8
        Xs = (X - self.mu) / self.sd
        d2 = torch.cdist(Xs[:300], Xs[:300]) ** 2
        sigma = math.sqrt(d2[d2 > 0].median().item() / 2)
        self.W = torch.randn(X.shape[1], DIM_MAX, generator=g) / sigma
        self.b = torch.rand(DIM_MAX, generator=g) * 2 * math.pi
    def phi(self, X):
        return math.sqrt(2 / DIM_MAX) * torch.cos(((X - self.mu) / self.sd) @ self.W + self.b)


def _mediana(cls, gerador, checar=True):
    vals = [rodar(cls(), gerador, sem, checar) for sem in SEMENTES]
    return float(torch.tensor(vals).median())


def avaliar():
    print(f"{'tarefa':<12}{'baseline':>10}{'referência':>12}{'você':>8}{'s_t':>7}")
    s = []
    for nome, gen in TAREFAS.items():
        b = _mediana(_Identidade, gen, checar=False)
        r = max(_mediana(_RFF, gen, checar=False), b + 1e-3)
        try:
            m = _mediana(Submissao, gen); erro = ""
        except AssertionError as e:
            m, erro = b, f"   <- {e}"
        st = min(max((m - b) / (r - b), 0.0), 1.25); s.append(st)
        print(f"{nome:<12}{b:>10.3f}{r:>12.3f}{m:>8.3f}{st:>7.2f}{erro}")
    S = 100 * (0.7 * sum(s) / len(s) + 0.3 * min(s))
    print(f"\nESCORE S = {S:.1f}   (0 = baseline, 100 = referência, até 125 com bônus)")
    return S


if __name__ == "__main__":
    avaliar()
