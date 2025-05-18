from solution_exact import saved_nodes_count
from dataclasses import dataclass
from math import exp
import random



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


from collections import deque

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

def choisir_voisin(graph, s):
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

def choisir_voisin2(graph, s):
    tous_arcs = s[:]

    for arc in graph.list_arcs:
        if arc not in tous_arcs:
            tous_arcs.append(arc)
    i, j = random.sample(range(len(tous_arcs)), 2)
    voisin = tous_arcs[:]
    voisin[i], voisin[j] = voisin[j], voisin[i]

    return voisin

def recuit_simule(graph, s, T, option=1):
    solution_max = s[:]
    valeur_max = saved_nodes_count(graph, solution_max)
    valeur = valeur_max
    tours = 0
    while T > 1:
        max_sans_amelioration = 60
        compteur_iter = 0
        while compteur_iter < max_sans_amelioration:
            compteur_iter += 1
            if option == 1:
                s_prime = choisir_voisin(graph, s)
            else:
                s_prime = choisir_voisin2(graph, s)
            valeur_prime = saved_nodes_count(graph, s_prime)
            if valeur_prime > valeur_max:
                s = s_prime
                valeur = valeur_prime
                valeur_max = valeur_prime
                solution_max = s_prime[:]
                print("Amélioration trouvée : ", solution_max, " pour : ", valeur_max, "a ", compteur_iter)
                compteur_iter = 0  # reset si amélioration
                
            else:
                delta = valeur_prime - valeur
                proba = exp(delta / T)
                if random.random() < proba:
                    s = s_prime
                    valeur = valeur_prime
        T *= 0.99  # Refroidissement
        tours += 1
    #print("nombre de tours : " + str(tours))
    return solution_max, valeur_max


def test(n):
    
    for i in range(n):
        #txt = "dag/dag_"+str(i)+".txt"
        txt = "dag/dagTest.txt"
        graph = load_graph(txt)
        result1, total1 = greedy_initial_solution(graph.arcs)
        if len(result1) == 0: continue
        
        result3, total3 = recuit_simule(graph, result1, 50, option=1)
        result4, total4 = recuit_simule(graph, result1, 50, option=2)
        print("test sur le fichier :", txt, end=" ")
        
        print("résultat de algo facile :", result1, " avec :", total1)

        if total4 > total3:
            print("permutation totale meilleurs :", result4, total4)
        elif total4 == total3:
            print("Pas de différence")
        else:
            print("swap d'acs meilleurs : ", result3, total3)
        print("")
        


test(1)