from dataclasses import dataclass
import itertools

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
                graph.arcs = {node: [] for node in range(int(nodes))}
                graph.infest = int(infest)
            else:
                parent, child = line.strip().split()
                graph.arcs[int(parent)].append(int(child))
                graph.list_arcs.append((int(parent), int(child)))
    return graph

def saved_nodes_count(graph, rm_arcs):
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

def best_order_arcs(graph):
    """
    Trouve l'ordre d'arcs qui minimise le nombre de noeuds infectés.
    """
    # On génère toutes les permutations des arcs
    # et on garde la meilleure
    best_count = -1
    best_order = None
    for perm in itertools.permutations(graph.list_arcs):
        count = saved_nodes_count(graph, perm)
        if count > best_count:
            best_count = count
            best_order = perm
        
    # On enlève les arcs qui ne sont pas nécessaires
    best_order = list(best_order)
    # Tant que la liste a plus d'un élément
    while len(best_order) > 1 and best_count == saved_nodes_count(graph, best_order[:-1]):
        best_order = best_order[:-1]

    return best_order

if __name__ == "__main__":
    # Exemple d'utilisation
    filename = "dag/dag_0.txt"
    graph = load_graph(filename)
    # print(f"{graph.arcs}")
    # print(saved_nodes_count(graph, [(2, 0), (4, 2)]))
    best = best_order_arcs(graph)
    print(best)
    print(saved_nodes_count(graph, best))
