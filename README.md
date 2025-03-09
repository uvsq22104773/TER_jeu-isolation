# TER - Jeu Isolation

## 1. Représentation du Problème :

### Le Graphe et les Données d’Entrée
- **Graph** : On considère un graphe connexe $G = (V,A,\tau, \lambda)$ où :
  - $V$ est l'ensemble des sommets.
  - $A$ est l’ensemble des arcs.
  - $\tau : A \rightarrow \mathbb{N}^*$ associe à chaque arc $[x, y]$ un entier non nul $\tau([x,y])$ qui représente le délai en nombre d’étapes nécessaire à la propagation d’une infection de $x$ vers $y$.
  - $\lambda[i][j] = 1$ si l'arc peut être supprimé,
  - $\lambda[i][j] = 0$ sinon.
- **Représentation** : Le graphe est fourni sous forme d’une matrice d’adjacence $M$ où :
  - $M[i][j] = \tau([i,j])$ si l’arc $[i,j]$ existe,
  - $M[i][j] = 0$ sinon.
- **Source de Propagation** : Un sommet $r \in V$ est le point de départ de l’infection.

### Dynamique de Propagation
- **Initialisation** :
  - À l’étape $0$ , seul $r$ est infecté.
  - On définit pour chaque sommet $v \in V$ une variable $T(v)$ qui représente l’instant (étape) auquel $v$ devient infecté. On initialise :
    - $T(r) = 0$
    - $T(v) = +\infty$ pour tout $v \neq r$.
- **Propagation** :
  - Si un sommet $x$ est infecté à l’étape $T(x)=t$, alors, pour tout arc $[x,y]$ toujours actif (non coupé) menant vers un sommet non infecté $y$, le sommet $y$ sera infecté à l’étape : 
  <br> $T(y) = t + \tau([x,y])$ 
  <br> *à condition que l’arc* $[x,y]$ *n’ait pas été supprimé avant l’instant* $t + \tau([x,y])$*.*

### Interventions (Suppression d’Arcs)
- **Principe** : À chaque étape temporelle (discrète), on peut supprimer un arc du graphe si $\lambda = 1$, ce qui revient à « fermer un canal » et empêcher ainsi l’infection par ce lien.
- **Séquence d’Actions** :
  - On cherche à établir une liste $L$ d’arcs à supprimer, où l’ordre dans $L$ correspond à l’ordre chronologique des suppressions.
  - L’intervention sur un arc $[x,y]$ doit être réalisée avant que l’infection ne se propage par ce lien, c’est-à-dire avant l’instant critique :
  <br> $t_{\text{crit}}([x,y]) = T(x) + \tau([x,y])$
- **Contrainte d’Intervention** :
  - À chaque étape $t$, une seule suppression est autorisée.
  - On ne peut supprimer un arc $[x,y]$ que si l’instant actuel $t$ est strictement inférieur à $t_{\text{crit}}([x,y])$.

### Objectif
L’objectif est de maximiser le nombre de sommets non infectés à la fin du processus de propagation, ce qui revient à interrompre la propagation le plus tôt possible. En pratique, cela se traduit par la suppression d’arcs « critiques » qui, une fois coupés, empêchent l’infection de se propager vers des sous-graphes entiers.
