from solution_exact import saved_nodes_count
from dataclasses import dataclass
import matplotlib.pyplot as plt
from collections import defaultdict
from math import exp
import heapq
import random
from typing import List, Tuple, Dict, Set
import time
from collections import deque

@dataclass
class Graph:
    infest : int
    arcs : dict
    list_arcs : list

def load_graph(filename):
    """
    Charge un graphe à partir d'un fichier texte.
    Le fichier doit contenir une liste d'arêtes, chaque ligne représentant
    une arête sous la forme 'noeud1 noeud2'.
    """
    graph = Graph(0, {}, [])
    firstline = True
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if firstline:
                firstline = False
                parts = line.strip().split()
                nodes, infest = parts[0], parts[1]
                graph.arcs = {node: [] for node in range(1, int(nodes)+1)}
                graph.infest = int(infest)
            else:
                parts = line.strip().split()
                parent, child = parts[0], parts[1]
                graph.arcs[int(parent)].append(int(child))
                graph.list_arcs.append((int(parent), int(child)))
    return graph

# Fonction de calcul récursif du nombre de sommets dans un sous-arbre
def calcul(tree, sommet):
    if sommet not in tree:
        return 1
    return 1 + sum(calcul(tree, fils) for fils in tree[sommet])

def plusGrandNombreDeFilsPuisGrandSousArbre(tree, ls=None, res=None):
    if ls is None:
        ls = [1]
    if res is None:
        res = []
    maxGlobal = 0
    areteGlobale = None
    aTraiter = []

    for s in ls:
        maxFils = 0
        maxSommet = 0
        arete = None

        if s not in tree:
            continue

        for fs in tree[s]:
            aTraiter.append(fs)
            nbFils = len(tree[fs]) if fs in tree else 0
            nbSommet = calcul(tree, fs)

            if nbFils > maxFils:
                maxFils = nbFils
                maxSommet = nbSommet
                arete = (s, fs)
            elif nbFils == maxFils and nbSommet > maxSommet:
                maxFils = nbFils
                maxSommet = nbSommet
                arete = (s, fs)

        if maxSommet > maxGlobal:
            maxGlobal = maxSommet
            areteGlobale = arete

    if areteGlobale:
        res.append(areteGlobale)
        if areteGlobale[1] in aTraiter:
            aTraiter.remove(areteGlobale[1])

    if not aTraiter:
        return res, maxGlobal
    else:
        sub_res, sub_max = plusGrandNombreDeFilsPuisGrandSousArbre(tree, aTraiter, res)
        return sub_res, maxGlobal + sub_max

def plusGrandSousArbreFirst(tree, ls=None, res=None):
    if ls is None:
        ls = [1]
    if res is None:
        res = []
    maxGlobal = 0
    arêteGlobal = None
    aTraiter = []

    for s in ls:
        if s not in tree:
            continue

        max_local = 0
        arête = None

        for fs in tree[s]:
            aTraiter.append(fs)
            nb = calcul(tree, fs)
            if nb > max_local:
                max_local = nb
                arête = (s, fs)

        if max_local > maxGlobal:
            maxGlobal = max_local
            arêteGlobal = arête

    if arêteGlobal:
        res.append(arêteGlobal)
        if arêteGlobal[1] in aTraiter:
            aTraiter.remove(arêteGlobal[1])

    if not aTraiter:
        return res, maxGlobal
    else:
        sub_res, sub_total = plusGrandSousArbreFirst(tree, aTraiter, res)
        return sub_res, maxGlobal + sub_total
    
def plusGrandSousArbrePuisFils(tree, ls=None, res=None):
    if ls is None:
        ls = [1]
    if res is None:
        res = []
    maxGlobal = 0
    areteGlobale = None
    aTraiter = []

    for s in ls:
        maxFils = 0
        maxSommet = 0
        arete = None

        if s not in tree:
            continue

        for fs in tree[s]:
            aTraiter.append(fs)
            nbFils = len(tree[fs]) if fs in tree else 0
            nbSommet = calcul(tree, fs)

            if nbSommet > maxSommet:
                maxFils = nbFils
                maxSommet = nbSommet
                arete = (s, fs)
            elif nbSommet == maxSommet and nbFils > maxFils:
                maxFils = nbFils
                maxSommet = nbSommet
                arete = (s, fs)

        if maxSommet > maxGlobal:
            maxGlobal = maxSommet
            areteGlobale = arete

    if areteGlobale:
        res.append(areteGlobale)
        if areteGlobale[1] in aTraiter:
            aTraiter.remove(areteGlobale[1])

    if not aTraiter:
        return res, maxGlobal
    else:
        sub_res, sub_max = plusGrandSousArbrePuisFils(tree, aTraiter, res)
        return sub_res, maxGlobal + sub_max


def get_descendants(graph, start, visited=None):
    if visited is None:
        visited = set()
    for neighbor in graph.get(start, []):
        if neighbor not in visited:
            visited.add(neighbor)
            get_descendants(graph, neighbor, visited)
    return visited

def filter_edges(graph, edges_to_remove):
    # Trouver tous les descendants des arêtes à supprimer
    nodes_to_exclude = set()
    for src, dst in edges_to_remove:
        nodes_to_exclude.add(dst)
        descendants = get_descendants(graph, dst)
        nodes_to_exclude.update(descendants)
    
    # Construire la liste finale des arêtes
    remaining_edges = []
    for src, neighbors in graph.items():
        for dst in neighbors:
            if src in nodes_to_exclude or dst in nodes_to_exclude:
                continue
            remaining_edges.append((src, dst))
    return remaining_edges


def choisir_voisin_arbre(graph, s):
    voisin = s[:]

    voisin.remove(random.choice(voisin))
    arcs_to_choose = filter_edges(graph.arcs, voisin)
    arcs_to_choose = graph.list_arcs
   
    if arcs_to_choose == []:
        return voisin
    arc = random.choice(arcs_to_choose)
    while arc in voisin:
        arc = random.choice(arcs_to_choose)
    voisin.append(arc)

    return voisin

def choisir_voisin_dag1(graph, s):
    voisin = s[:]

    voisin.remove(random.choice(voisin))
    arcs_to_choose = graph.list_arcs
   
    if arcs_to_choose == []:
        return voisin
    arc = random.choice(arcs_to_choose)
    while arc in voisin:
        arc = random.choice(arcs_to_choose)
    voisin.append(arc)

    return voisin

def choisir_voisin_dag2(graph, s):
    tous_arcs = s[:]

    for arc in graph.list_arcs:
        if arc not in tous_arcs:
            tous_arcs.append(arc)
    i, j = random.sample(range(len(tous_arcs)), 2)
    voisin = tous_arcs[:]
    voisin[i], voisin[j] = voisin[j], voisin[i]

    return voisin

def compute_reachability_counts(graph):
    """
    graph: dict node -> list of successors
    Retourne reach[v] = taille du sous-DAG accessible depuis v (incluant v).
    """
    # 1. Tri topologique
    in_deg = {u: 0 for u in graph}
    for u in graph:
        for v in graph[u]:
            in_deg[v] += 1
    queue = deque(u for u in graph if in_deg[u] == 0)
    topo = []
    while queue:
        u = queue.popleft()
        topo.append(u)
        for v in graph[u]:
            in_deg[v] -= 1
            if in_deg[v] == 0:
                queue.append(v)
    # 2. DP en ordre inverse
    reach = {u: 1 for u in graph}  # compte soi-même
    for u in reversed(topo):
        for v in graph[u]:
            reach[u] += reach[v]
    return reach

def greedy_initial_solution(graph):
    """
    Simule la propagation/infection et choisit à chaque étape
    l'arc à couper qui sauve le plus grand sous-DAG.
    
    graph: dict node -> list of successors (mutable copy)
    start: nœud initialement infecté
    """
    # Copie des arcs
    succ = {u: set(vs) for u, vs in graph.items()}
    reach = compute_reachability_counts(graph)
    
    infected = {1}
    solution = []  # liste des arcs coupés
    
    while True:
        # 1️⃣ calcul du frontier
        frontier = []
        for u in list(infected):
            for v in succ[u]:
                if v not in infected:
                    frontier.append((u, v))
        if not frontier:
            break
        
        # 2️⃣ choisir l'arc qui sauve le plus grand sous-DAG
        best_arc = max(frontier, key=lambda uv: reach[uv[1]])
        solution.append(best_arc)
        succ[best_arc[0]].remove(best_arc[1])
        
        # 3️⃣ propager infection d'un pas
        new_inf = set()
        for u in infected:
            for v in succ[u]:
                if v not in infected:
                    new_inf.add(v)
        if not new_inf:
            break
        infected |= new_inf
    
    # renvoyer la liste des arcs coupés et le nombre de sauvés
    saved = len(graph) - len(infected)
    return solution, saved

def recuit_simule(graph, s, T, option=1):
    solution_max = s[:]
    valeur_max = saved_nodes_count(graph, solution_max)
    valeur = valeur_max
    tours = 0
    amelioration = []
    while T > 1:
        max_sans_amelioration = 60
        compteur_iter = 0
        while compteur_iter < max_sans_amelioration:
            tours += 1
            compteur_iter += 1
            if option == 0:
                s_prime = choisir_voisin_arbre(graph, s)
            if option == 1:
                s_prime = choisir_voisin_dag1(graph, s)
            else:
                s_prime = choisir_voisin_dag2(graph, s)
            valeur_prime = saved_nodes_count(graph, s_prime)
            if valeur_prime > valeur_max:
                amelioration.append((valeur_prime - valeur_max, tours))
                s = s_prime
                valeur = valeur_prime
                valeur_max = valeur_prime
                solution_max = s_prime[:]
                print("Amélioration trouvée : ", solution_max, " pour : ", valeur_max, "au bout de  ", compteur_iter, " itérations pour un total de ", tours)
                compteur_iter = 0  # reset si amélioration
                
            else:
                delta = valeur_prime - valeur
                proba = exp(delta / T)
                if random.random() < proba:
                    s = s_prime
                    valeur = valeur_prime
        T *= 0.99  # Refroidissement
    #print("nombre de tours : " + str(tours))
    return solution_max, valeur_max, amelioration

def score_chain_pas_opti(graph, rm_arcs):
    """
    Compte le nombre de noeuds sauver dans un graph
    à partir d'une suite d'arcs.
    """
    infected_nodes = set()
    infected_nodes.add(graph.infest)
    copy_arcs = {}
    for node in graph.arcs:
        copy_arcs[node] = graph.arcs[node].copy()
    for arc in rm_arcs:
        copy_arcs[arc[0]].remove(arc[1])
        for node in list(infected_nodes):
            infected_nodes.update(copy_arcs[node])
    
    while True:
        new_infected = set()
        for node in infected_nodes:
            new_infected.update(copy_arcs[node])
        if not new_infected - infected_nodes:
            break
        infected_nodes.update(new_infected)

    return len(graph.arcs) - len(infected_nodes)

from collections import deque

def score_chain(graph, chain):
    """
    Simule la propagation temporelle avec suppression séquentielle d'arcs.
    À chaque étape :
      1) on supprime l'arc de la chaîne,
      2) on propage l'infection d'un pas (tau=1 pour chaque arc).
    """
    # Ensemble des arcs à supprimer au fur et à mesure
    removed = set()
    # Sommets infectés et frontière actuelle
    infected = {graph.infest}
    frontier = {graph.infest}

    for u, v in chain:
        # 1) suppression de l'arc
        removed.add((u, v))

        # 2) propagation d'un pas
        new_frontier = set()
        for x in frontier:
            for y in graph.arcs.get(x, []):
                if (x, y) not in removed and y not in infected:
                    infected.add(y)
                    new_frontier.add(y)
        frontier = new_frontier

    # Optionnel : propagation finale jusqu'à stabilisation
    queue = deque(frontier)
    while queue:
        x = queue.popleft()
        for y in graph.arcs.get(x, []):
            if (x, y) not in removed and y not in infected:
                infected.add(y)
                queue.append(y)

    # Calcul des sommets sauvés
    all_nodes = set(graph.arcs.keys()) | {v for children in graph.arcs.values() for v in children}
    return len(all_nodes) - len(infected)

def randomized_epsilon_greedy(graph: Graph, epsilon: float, max_iter: int):
    """
    ε-greedy avec tas binaire persistant pour exploitation en O(log m).
    Retourne la meilleure chaîne (permutation d’arcs) et son score.
    """
    resultat = []
    j_max = len(graph.list_arcs)
    # Compteurs par position et par arc
    N = [defaultdict(int) for _ in range(j_max)]
    W = [defaultdict(int) for _ in range(j_max)]
    
    # Chaîne initiale aléatoire
    best_chain = random.sample(graph.list_arcs, j_max)
    best_score = score_chain(graph, best_chain)
    resultat.append(best_score)
    
    # Initialisation des tas (ratio initial = 0)
    heaps = []
    for j in range(j_max):
        heap = [(0.0, arc) for arc in graph.list_arcs]
        heapq.heapify(heap)
        heaps.append(heap)
    for _ in range(max_iter):
        chain: List[Tuple[int, int]] = []
        used: Set[Tuple[int, int]] = set()
        for j in range(j_max):
            if len(used) == j_max:
                break
            
            # Exploration vs exploitation
            if random.random() < epsilon:
                # exploration
                choices = [a for a in graph.list_arcs if a not in used]
                arc = random.choice(choices)
            else:
                heap = heaps[j]
                while heap:
                    neg_ratio, cand = heap[0]
                    current = W[j][cand] / (N[j][cand] + 1)

                    if cand in used:
                        heapq.heappop(heap)
                        continue

                    if -neg_ratio != current:
                        heapq.heapreplace(heap, (-current, cand))
                        continue

                    arc = cand
                    break
                else:
                    # pas de candidat (sauvegarde)
                    remaining = [a for a in graph.list_arcs if a not in used]
                    arc = random.choice(remaining)

            chain.append(arc)
            used.add(arc)
        # Évaluation et rétro-propagation des scores
        sc = score_chain(graph, chain)
        resultat.append(sc)
        for k, a in enumerate(chain):
            N[k][a] += 1
            W[k][a] += sc
            # Mettre à jour le tas avec le nouveau ratio
            new_ratio = W[k][a] / (N[k][a] + 1)
            heapq.heappush(heaps[k], (-new_ratio, a))
        
        if sc > best_score:
            best_score = sc
            best_chain = list(chain)
    
    return best_chain, best_score, resultat

def tabu_search(graph: Graph):
    """
    Tabu Search pour l'isolation de propagation sur un DAG.
    Retourne la meilleure suite d'arcs à supprimer.
    """
    # --- Initialisation ---
    # on choisit une chaîne de base (par exemple aléatoire)
    taille = len(graph.list_arcs)
    chaine_max = random.sample(graph.list_arcs, taille)
    max_score = score_chain(graph, chaine_max)
    
    resultat = []
    chaine = list(chaine_max)
    note = max_score
    resultat.append(max_score)
    
    tabu = deque()
    tabu.append(tuple(chaine))
    
    tours = 0
    occurence_max = taille
    occurence = 0
    j = 0  # indice pivot (0-based)
    
    # --- Boucle principale ---
    while occurence < occurence_max:
        # on essaie tous les swaps (i, j)
        for i in range(taille):
            if i == j:
                continue
            tmp = chaine.copy()
            # échange des positions i et j
            tmp[i], tmp[j] = tmp[j], tmp[i]
            tmp_tup = tuple(tmp)
            tmp_score = score_chain(graph, tmp)
            
            # si non tabou ou s'il améliore le meilleur global
            if tmp_tup not in tabu or tmp_score > max_score:
                # acceptation si améliore la note courante
                if tmp_score > note:
                    note = tmp_score
                    chaine = tmp
        # fin du balayage sur j fixe
        
        occurence += 1
        tours += 1
        resultat.append(note)
        # si on améliore le meilleur global, on met à jour et on réinitialise
        if note > max_score:
            max_score = note
            chaine_max = chaine.copy()
            occurence = 0
        
        # mise à jour de la liste tabou
        tabu.append(tuple(chaine))
        if len(tabu) > 20:
            tabu.popleft()
        
        # avancer le pivot j
        j = (j + 1) % taille
    
    return chaine_max, max_score, tours, resultat

def random_isolation(graph: Graph, occur_max: int = 100):
    """
    Algorithme purement aléatoire pour isoler la propagation dans un arbre/DAG.
    À chaque itération, on construit une chaîne de coupures aléatoire, puis on garde
    la meilleure sur 'occur_max' essais.
    """
    # Chaîne de base (premier essai)
    def build_random_chain():
        chain = []
        tmp_chain = graph.list_arcs.copy()
        while len(tmp_chain) > 0:
            arc = random.choice(tmp_chain)
            tmp_chain.remove(arc)
            chain.append(arc)
        return chain

    resultat = []
    best_chain = build_random_chain()
    best_score = score_chain(graph, best_chain)
    resultat.append(best_score)

    for _ in range(occur_max):
        chain = build_random_chain()
        sc = score_chain(graph, chain)
        resultat.append(sc)
        if sc > best_score:
            best_score = sc
            best_chain = chain

    return best_chain, best_score, resultat

# graph = load_graph("tree/arbre_0.txt")
graph = load_graph("dag/dag_0.txt")

def genereGraph(listIter, listBase):
    max_taille = 0
    for elem in listIter:
        if len(elem) > max_taille:
            max_taille = len(elem)
    listRes = [0]*max_taille
    listNb = [0]*max_taille
    

# print(graph.infest)
# print(graph.arcs)
# print(graph.list_arcs)
# result, total = plusGrandSousArbreFirst(graph.arcs, [45])
print("Recuit")
start = time.time()
result1, total1 = greedy_initial_solution(graph.arcs)
print("résultat de algo facile :", result1, " avec :", total1)
result3, total3, ame1 = recuit_simule(graph, result1, 50, option=1)
result4, total4, ame2 = recuit_simule(graph, graph.list_arcs, 50, option=2)
end = time.time()

# print("Arêtes supprimées :", resultat)
if total4 > total3:
    print("permutation totale meilleurs :", result4, total4)
elif total4 == total3:
    print("Pas de différence")
else:
    print("swap d'acs meilleurs : ", result3, total3)
print("")
print(f"Temps d'éxecution : {end - start}s")

print("\nTabou")
start = time.time()
resultat, total, tours, resultat4 = tabu_search(graph)
end = time.time()

print("Nombre d'occurence total =", tours)
# print("Arêtes supprimées :", resultat)
print("Nombre de sommets sauvés :", total, score_chain_pas_opti(graph, resultat))
print(f"Temps d'éxecution : {end - start}s")

print("\nRandom")
start = time.time()
result, score, resultat2 = random_isolation(graph, tours)
end = time.time()

# print("Arêtes supprimées :", result)
print("Nombre de sommets sauvés :", score, score_chain_pas_opti(graph, result))
print(f"Temps d'éxecution : {end - start}s")

print("\nRandom epsilon greedy")
start = time.time()
result, score, resultat3 = randomized_epsilon_greedy(graph, 0.3, tours)
end = time.time()

# print("Arêtes supprimées :", result)
print("Nombre de sommets sauvés :", score, score_chain_pas_opti(graph, result))
print(f"Temps d'éxecution : {end - start}s")

x = list(range(1, len(resultat4) + 1))

# plt.plot(x, resultat1, label='Recuit')
# plt.plot(x, resultat2, label='Random')
# plt.plot(x, resultat3, label='random epsilon greedy')
plt.plot(x, resultat4, label='Tabou')

plt.xlabel('Numéro de la simulation')
plt.ylabel('Valeur sauvé')
plt.title('Graphique de 3 courbes')
plt.legend()
plt.grid(True)
plt.show()
