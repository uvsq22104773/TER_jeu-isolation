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


def choisir_voisin(graph, s):
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

def recuit_simule(graph, s, T):
    solution_max = s[:]
    valeur_max = saved_nodes_count(graph, solution_max)
    valeur = valeur_max
    tours = 0
    while T > 1:
        max_sans_amelioration = 60
        compteur_iter = 0
        while compteur_iter < max_sans_amelioration:
            compteur_iter += 1
            s_prime = choisir_voisin(graph, s)
            valeur_prime = saved_nodes_count(graph, s_prime)
            if valeur_prime > valeur_max:
                s = s_prime
                valeur = valeur_prime
                valeur_max = valeur_prime
                solution_max = s_prime[:]
                #print("Amélioration trouvée : ", solution_max, " pour : ", valeur_max, "a ", compteur_iter)
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
        txt = "tree/arbre_"+str(i)+".txt"
        graph = load_graph(txt)
        result1, total1 = plusGrandSousArbreFirst(graph.arcs)
        result2, total2 = plusGrandSousArbrePuisFils(graph.arcs)
        result3, total3 = plusGrandNombreDeFilsPuisGrandSousArbre(graph.arcs)
        
        options = [(total1, result1),(total2, result2),(total3, result3),]
        _, worst_result = min(options, key=lambda x: x[0])
        result4, total4 = recuit_simule(graph, worst_result, 50)
        print("test sur le fichier :", txt, end=" ")
        
        best_total, best_result = max(options, key=lambda x: x[0])
        print("résultat max des algos facile :", best_total, " avec :", best_result)

        if total4 > best_total:
            print("Recuit meilleur que les autres algos : ", result4, total4)
        elif total4 == best_total:
            print("Pas de différence")
        else:
            print("Recuit moins bon que les autres algos : ", result4, total4)
        print("")


test(30)