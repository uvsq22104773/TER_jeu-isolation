from dataclasses import dataclass
import itertools
import sys
import multiprocessing
import itertools
from collections import deque

# Structure pour un graph
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
                nodes, infest = line.strip().split()
                graph.arcs = {node: [] for node in range(1, int(nodes)+1)}
                graph.infest = int(infest)
            else:
                parent, child = line.strip().split()
                graph.arcs[int(parent)].append(int(child))
                graph.list_arcs.append((int(parent), int(child)))
    return graph

def saved_nodes_count(graph, chain):
    """
    Compte le nombre de noeuds sauver dans un graph
    à partir d'une suite d'arcs.
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

def factorielle(n):
    """
    Calcule la factorielle de n.
    """
    resultat = 1
    for i in range(1, n + 1):
        resultat *= i
    return resultat

def _evaluate_perm(args):
    graph_data, perm = args
    arcs_dict, infest = graph_data
    graph_copy = Graph(infest, {k: v.copy() for k, v in arcs_dict.items()}, list(perm))
    return (saved_nodes_count(graph_copy, perm), perm)

def best_order_arcs_multithreading(graph):
    """
    Trouve l'ordre d'arcs qui minimise le nombre de noeuds infectés.
    Version optimisée sans surcharge mémoire (pas de stockage global des permutations).
    """
    graph_data = (graph.arcs, graph.infest)
    total_perms = factorielle(len(graph.list_arcs))
    batch_size = 1000

    if total_perms > 10000:
        diviseur = 1000
    elif total_perms < 100:
        diviseur = 1
    else:
        diviseur = 10

    best_count = -1
    best_order = None
    i = 0

    # Créer un générateur de permutations
    perm_generator = itertools.permutations(graph.list_arcs)

    with multiprocessing.Pool() as pool:
        while True:
            # Préparer un batch de permutations
            batch = list(itertools.islice(perm_generator, batch_size))
            if not batch:
                break

            # Appliquer la fonction en parallèle sur le batch
            results = pool.map(_evaluate_perm, ((graph_data, perm) for perm in batch))

            for count, perm in results:
                i += 1
                if i % total_perms//diviseur == 0 or i == total_perms:
                    print(f"\rProgression : {(i / total_perms) * 100:.1f}%", end="")
                if count > best_count:
                    best_count = count
                    best_order = perm

    print("\rProgression : 100%")
    return best_order

def ajouter_au_debut(fichier, texte_a_ajouter):
    with open(fichier, 'r') as f:
        lignes = f.readlines()

    # Modifier la première ligne
    if lignes:
        lignes[0] = lignes[0][:len(lignes[0])-1] + " " + texte_a_ajouter + "\n"
    else:
        lignes = [texte_a_ajouter + "\n"]

    with open(fichier, 'w') as f:
        f.writelines(lignes)

'''
Ajout sur la première ligne du fichier 
la suite d'arcs optimale et le 
nombre de noeuds sauvés.
'''

def solution_for_n_tree(n, fn):
    """
    Trouve la solution pour n arbres orientés aléatoires.
    """
    result = []
    print(f"\rProgression globale : 0.0%")
    for i in range(n):
        sys.stdout.write("\033[F")
        print(f"\rProgression globale : {(i/n)*100:.1f}%")
        filename = f"{fn}{i}.txt"
        graph = load_graph(filename)
        order = best_order_arcs_multithreading(graph)
        count = saved_nodes_count(graph, order)
        add_str = "[" + " ".join(map(str, order)) + "] " + str(count)
        ajouter_au_debut(filename, add_str)
    print()

if __name__ == "__main__":
    # Exemple d'utilisation
    # print(f"{graph.arcs}")
    # print(saved_nodes_count(graph, [(2, 0), (4, 2)]))
    # solution_for_n_tree(1, "tree/arbre_")
    print("arbre_6.txt")
    graph = load_graph("tree/arbre_6.txt")
    best_order = best_order_arcs_multithreading(graph)
    print(best_order)
    print(saved_nodes_count(graph, best_order))
    """
    print("\n")

    print("arbre_0.txt - simple")
    graph = load_graph("tree/arbre_0.txt")
    best_order = best_order_arcs(graph)
    print(best_order)
    print(saved_nodes_count(graph, best_order))
    """
    print("\n")
    print("arbre_1.txt - multithreading")
    graph = load_graph("tree/arbre_1.txt")
    best_order = best_order_arcs_multithreading(graph)
    print(best_order)
    print(saved_nodes_count(graph, best_order))
