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

def plusGrandNombreDeFilsPuisGrandSousArbre(tree, ls = [1], res = []):
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

def plusGrandSousArbreFirst(tree, ls = [1], res = []):
    maxGlobal = 0
    arêteGlobal = None
    aTraiter = []

    for s in ls:
        if s not in tree:
            continue  # Pas de fils

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

def choisir_voisin(graph, s):
    voisin = s[:]
    voisin.remove(random.choice(voisin))
    arc = random.choice(graph.list_arcs)
    while arc in voisin:
        arc = random.choice(graph.list_arcs)
    voisin.append(arc)
    return voisin

def recuit_simule(graph, s, T):
    solution_max = s[:]
    valeur_max = saved_nodes_count(graph, solution_max)
    tours = 0
    while T > 1:
        s_prime = choisir_voisin(graph, s)
        valeur = saved_nodes_count(graph, s_prime)

        # Accepter si meilleure ou avec probabilité
        if valeur >= valeur_max:
            s = s_prime
            valeur_max = valeur
            solution_max = s_prime[:]
        else:
            delta = valeur - saved_nodes_count(graph, s)
            proba = exp(delta / T)
            if random.random() < proba:
                s = s_prime

        T *= 0.99  # Refroidissement
        tours += 1
    print("nombre de tours : " + str(tours))
    return solution_max, valeur_max



graph = load_graph("tree/arbre_0.txt")


result, total = plusGrandSousArbreFirst(graph.arcs)


print("Arêtes supprimées :", result)
print("Nombre de sommets sauvés :", total)

resultat, total = recuit_simule(graph, result, 100)
print(resultat, total)

def test():
    for i in range(1):
        txt = "tree/arbre_"+str(i)+".txt"
        graph = load_graph(txt)
        result1, total1 = plusGrandSousArbreFirst(graph.arcs)
        result2, total2 = recuit_simule(graph, result1, 100)
        if result1 != result2:
            print("différence : ")
            print(result1, total1)
            print(result2, total2)
