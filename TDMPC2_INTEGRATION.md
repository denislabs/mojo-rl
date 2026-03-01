# Intégration de TDMPC2 dans mojo-rl

> Analyse des composants à ajouter pour implémenter TD-MPC2
> (Hansen et al., 2023 — "TD-MPC2: Scalable, Robust World Models for Continuous Control")

---

## Vue d'ensemble de TDMPC2

TDMPC2 est un algorithme **model-based RL** qui apprend simultanément :
- un **encodeur** : observation → espace latent `z`
- un **modèle de dynamiques** : `(z, a) → z'` (prédiction dans l'espace latent)
- un **modèle de récompense** : `(z, a) → r` (distributional)
- un **modèle de terminaison** : `(z) → done`
- une **politique** : `z → (μ, σ)` (Gaussienne, sert de prior pour MPPI)
- un **ensemble de Q-fonctions** : `(z, a) → Q` (distributional)

À chaque pas de temps, TDMPC2 utilise **MPPI** (Model Predictive Path Integral) pour planifier dans l'espace latent sur un horizon H, en utilisant le world model pour évaluer les trajectoires candidates.

---

## Ce qui est déjà dans mojo-rl ✅

### Couches neurales (`deep_rl/model/`)
| Composant | Fichier | Usage TDMPC2 |
|-----------|---------|--------------|
| `Linear` | `linear.mojo` | Couche de base de toutes les MLPs |
| `LayerNorm` | `layer_norm.mojo` | Dans `NormedLinear` |
| `Dropout` | `dropout.mojo` | Dans `NormedLinear` (Q-functions) |
| `ReLU`, `Tanh`, `Sigmoid`, `Softmax` | fichiers dédiés | Activations de base |
| `StochasticActor` | `stochastic_actor.mojo` | Base pour la politique |
| `Sequential` | `sequential.mojo` | Composition des MLPs |

### Optimiseurs (`deep_rl/optimizer/`)
| Optimiseur | Usage TDMPC2 |
|------------|--------------|
| `Adam` | World model + politique |
| `AdamW` | Alternative pour world model |

### Pertes (`deep_rl/loss/`)
| Perte | Usage TDMPC2 |
|-------|--------------|
| `MSE` | Consistency loss (prédiction dynamiques) |
| `CrossEntropy` | Base pour soft cross-entropy (reward/value) |

### Agents et infrastructure
| Composant | Usage TDMPC2 |
|-----------|--------------|
| `ReplayBuffer` | Stockage des transitions |
| `SAC` | Architecture de référence (twin Q, stochastic actor) |
| `Network` (training) | Wrapper params + modèle |
| Checkpoint system | Sauvegarde/chargement |

---

## Ce qui manque — À créer 🔨

### 1. Nouvelles couches neurales

#### `deep_rl/model/mish.mojo` — Activation Mish
```
Mish(x) = x * tanh(softplus(x)) = x * tanh(log(1 + e^x))
```
C'est l'activation par défaut de `NormedLinear` dans TDMPC2.
Gradient : `f'(x) = tanh(sp) + x * σ(x) * (2 - tanh²(sp))` où `sp = softplus(x)`.

**Complexité** : Faible — simple activation, forward + backward straightforward.

---

#### `deep_rl/model/simnorm.mojo` — Simplicial Normalization
```
SimNorm(dim)(x) :
  1. Reshape x: [..., D] → [..., D/dim, dim]
  2. Softmax sur la dernière dimension
  3. Reshape retour → [..., D]
```
Utilisée dans le **modèle de dynamiques** pour stabiliser l'espace latent
(remplace LayerNorm dans la couche de sortie du dynamics model).

**Paramètres** : `dim` (taille des groupes, typiquement 8).
**Pas de paramètres appris** — normalization pure.
**Complexité** : Faible — déjà Softmax dans mojo-rl, c'est un reshape + softmax.

---

#### `deep_rl/model/normed_linear.mojo` — NormedLinear block
```
NormedLinear(in, out, dropout=0., act=Mish):
  Linear(in, out) → Dropout → LayerNorm(out) → Mish
```
C'est le bloc de base de **toutes** les MLPs de TDMPC2 (sauf la dernière couche).
La couche finale utilise `Linear` pur, optionnellement avec `SimNorm` (dynamics).

**Complexité** : Faible — composition de couches existantes.
Note : nécessite Mish d'abord.

---

### 2. Nouvelles fonctions de perte

#### `deep_rl/loss/two_hot.mojo` — Two-hot encoding
TDMPC2 utilise le RL distributional : les récompenses et valeurs sont représentées
comme des distributions sur `num_bins` bins équidistants dans `[v_min, v_max]`.

```
two_hot(x, bins) :
  - Trouver les deux bins adjacents à x : bins[k] ≤ x < bins[k+1]
  - Interpolation linéaire :
    target[k]   = (bins[k+1] - x) / (bins[k+1] - bins[k])
    target[k+1] = (x - bins[k])   / (bins[k+1] - bins[k])
    target[i]   = 0 ailleurs
```

Produit un vecteur one-hot "flou" de dimension `num_bins`.
Typiquement : `num_bins=101`, `v_min=-10`, `v_max=10`.

**Complexité** : Moyenne — calcul vectoriel, attention aux bornes.

---

#### `deep_rl/loss/soft_cross_entropy.mojo` — Soft cross-entropy
```
L = -Σ_i target_i * log(softmax(logits)_i)
  = -Σ_i target_i * log_softmax(logits)_i
```
Où `target` est le two-hot vector (soft, pas one-hot pur).
C'est la perte utilisée pour **reward** et **Q-values** (distributional).

**Complexité** : Faible — déjà cross_entropy.mojo dans le projet, à étendre.

---

### 3. Replay Buffer multi-step

#### `deep_rl/replay/sequence_replay_buffer.mojo` — Stockage de séquences
La perte de consistance de TDMPC2 nécessite de **dérouler le world model sur H pas**.
Le replay buffer doit donc stocker et échantillonner des **séquences de transitions** de longueur `H+1`, pas des transitions individuelles.

```
Interface :
  add(obs, action, reward, done)   // Ajout continu
  sample_sequence[BATCH, H]() → (obs[H+1], actions[H], rewards[H], dones[H])
```

**Complexité** : Moyenne — buffer circulaire avec gestion des séquences (éviter
les coupures en fin d'épisode).

---

### 4. World Model

#### `deep_agents/tdmpc2/world_model.mojo` — Architecture complète

```
WorldModel[OBS_DIM, ACTION_DIM, LATENT_DIM, MLP_DIM, NUM_BINS, NUM_Q]:

  encoder:    MLP(OBS_DIM → LATENT_DIM)
              [NormedLinear(OBS_DIM, MLP_DIM), NormedLinear(MLP_DIM, LATENT_DIM)]

  dynamics:   MLP(LATENT_DIM + ACTION_DIM → LATENT_DIM)
              [NormedLinear(LATENT_DIM+ACTION_DIM, MLP_DIM),
               NormedLinear(MLP_DIM, LATENT_DIM),
               Linear(LATENT_DIM, LATENT_DIM) + SimNorm(8)]

  reward:     MLP(LATENT_DIM + ACTION_DIM → NUM_BINS)
              [NormedLinear(..., MLP_DIM), NormedLinear(MLP_DIM, MLP_DIM),
               Linear(MLP_DIM, NUM_BINS)]

  termination: MLP(LATENT_DIM → 1)
               [NormedLinear(...), NormedLinear(...), Linear → Sigmoid]

  policy:     MLP(LATENT_DIM → 2 * ACTION_DIM)  // mean + log_std
              [NormedLinear(...), NormedLinear(...), Linear]

  Q_ensemble: NUM_Q × MLP(LATENT_DIM + ACTION_DIM → NUM_BINS)
              [NormedLinear(dropout sur 1ère couche), NormedLinear, Linear]
```

Dimensions typiques (tâche single) : `LATENT_DIM=512`, `MLP_DIM=512`, `NUM_BINS=101`, `NUM_Q=5`.

**Complexité** : Élevée — la pièce centrale. Necessite toutes les couches ci-dessus.

---

### 5. Planificateur MPPI

#### `deep_agents/tdmpc2/mppi.mojo` — Model Predictive Path Integral

```
plan(z0, world_model, num_iterations, horizon, num_samples, num_pi_trajs, temperature):

  1. Initialisation :
     - Tirer num_pi_trajs trajectoires avec la politique apprise
     - mean[H, ACTION_DIM] = 0 (ou décalé du pas précédent)
     - std[H, ACTION_DIM] = 0.5

  2. Pour chaque itération :
     a. Générer candidates : actions = mean + std * noise, clipper [-1,1]
     b. Pour chaque candidat, dérouler le world model sur H pas :
        z_t+1 = dynamics(z_t, a_t)
        r_t   = reward(z_t, a_t)
        G     = Σ_t γ^t * r_t + γ^H * Q_min(z_H, π(z_H))
     c. Top-k elite par valeur G
     d. Softmax weights : w = exp(temperature * (G - max(G)))
     e. mean = Σ w_i * a_i (moyenne pondérée des élites)
     f. std  = sqrt(Σ w_i * (a_i - mean)^2)

  3. Sélection : Gumbel-softmax sur scores des élites → action_0
```

**Complexité** : Élevée — boucle de planification imbriquée, nombreux appels au world model.

---

### 6. Agent principal TDMPC2

#### `deep_agents/tdmpc2/tdmpc2.mojo` — Boucle d'entraînement

```
Hyperparamètres clés :
  H = 3          // horizon de planification
  γ = 0.99       // discount factor
  ρ = 0.5        // decay du poids temporel dans les pertes
  τ = 0.01       // soft update des target networks
  batch_size = 256
  learning_rate = 3e-4
  consistency_coef = 2.0
  reward_coef = 0.5
  value_coef = 0.1
  entropy_coef = 1e-4
```

**Boucle d'update (1 step) :**

```
1. Encoder la séquence : z_0 = encode(obs_0)

2. TD target (pour chaque t) :
   z_next = encode(obs_t+1)    // encodage sans gradient
   a_next ~ π(z_next)
   Q_target = r_t + γ * (1-done) * Q_min(z_next, a_next)
   Q_target_dist = two_hot(Q_target)

3. Latent rollout + accumulation des pertes :
   Pour t = 0..H-1 :
     z_pred_t+1 = dynamics(z_t, a_t)
     z_enc_t+1  = encode(obs_t+1)   // sg (stop gradient)

     L_consistency += ρ^t * MSE(z_pred, sg(z_enc))
     L_reward      += ρ^t * soft_CE(reward_pred(z_t, a_t), two_hot(r_t))
     L_value       += ρ^t * soft_CE(Q(z_t, a_t), Q_target_dist_t)

     z_t = z_pred_t+1  // continuer avec le prédit

4. Total loss :
   L = 2.0 * L_consistency + 0.5 * L_reward + 0.1 * L_value
   Backprop + Adam step (world model params)

5. Policy update :
   Pour t = 0..H-1 :
     a_pi ~ π(z_t)
     L_pi += -ρ^t * (Q_min(z_t, a_pi) + entropy_coef * H(π))
   Adam step (policy params seulement)

6. Soft update des target Q-networks :
   θ_target ← τ * θ + (1-τ) * θ_target
```

---

## Plan d'implémentation recommandé

### Phase 1 — Nouvelles couches (1-2 jours)
1. `mish.mojo` — activation
2. `simnorm.mojo` — normalisation
3. `normed_linear.mojo` — bloc de base

### Phase 2 — Pertes et replay (1-2 jours)
4. `soft_cross_entropy.mojo`
5. `two_hot.mojo`
6. `sequence_replay_buffer.mojo`

### Phase 3 — World Model (3-5 jours)
7. `world_model.mojo` avec forward/backward de toutes les têtes

### Phase 4 — Planning + Agent (3-5 jours)
8. `mppi.mojo` — planificateur
9. `tdmpc2.mojo` — agent complet

### Phase 5 — Validation (2-3 jours)
10. Test sur `PendulumEnv` (simple, référence)
11. Benchmarks sur `HopperEnv`, `HalfCheetahEnv`
12. Comparaison SAC vs TDMPC2 (sample efficiency)

---

## Points d'attention techniques

**Stop-gradient** : La consistency loss nécessite de stopper le gradient sur
l'encodage de `obs_t+1`. En Mojo, cela signifie appeler `encode()` en mode
inférence (sans cacher les activations) pour la cible.

**Ensemble de Q-networks** : TDMPC2 utilise 5 Q-networks et prend le minimum
de 2 tirés aléatoirement lors de la policy update. Implémenter via un tableau
de `Network[...]` et sous-échantillonnage.

**Distributional RL** : Les Q-fonctions sortent des logits sur `NUM_BINS=101`
valeurs. La valeur scalaire est `Σ_i softmax(logits)_i * bins_i`.

**Two optimiseurs** : World model (encoder + dynamics + reward + Q) et policy
ont des optimiseurs **séparés** avec un `enc_lr_scale` pour l'encodeur
(typiquement 0.3x le LR du reste).

**MPPI vectorisé** : La boucle MPPI évalue `num_samples=512` trajectoires.
L'idéal est de l'exécuter par batch sur GPU — potentiellement le gain
de performance le plus important versus l'implémentation Python de LeRobot.
