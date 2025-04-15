###############################################################################################################
## CONTENU DES FICHIERS GÉNÉRÉS :                                                                            ##
###############################################################################################################
## Chaque fichier contient la représentation d'un DAG ou d'un arbre orienté sous forme de liste d'adjacence. ##
## La première ligne contient le nombre de sommets et le sommet infecté.                                     ##
## Les lignes suivantes contiennent les arêtes sous la forme "u v" où u est le parent de v.                  ##
###############################################################################################################
## EXEMPLE DE CONTENU D'UN FICHIER :                                                                         ##
###############################################################################################################
## 5 3 ## (Le 5 correspond au nombre de sommets et le 3 au premier sommet infecté)                           ##
## 3 5 ## (À partir de la ligne 2, on trouve les arêtes, donc 3 est le parent de 5)                          ##
## 5 0 ##                                                                                                    ##
###############################################################################################################

import random
from pathlib import Path

def creer_arbre_oriente_racine_aleatoire(n):
    """
    Crée un arbre orienté (arbre enraciné) sans circuit avec une racine choisie aléatoirement.
    L'arbre est représenté sous forme de dictionnaire où chaque clé correspond à un nœud
    et la valeur associée est la liste de ses enfants.
    
    Pour construire l'arbre, on commence par mélanger la liste des nœuds.
    Le premier nœud devient la racine, et pour chaque nœud suivant, un parent est choisi
    aléatoirement parmi les nœuds déjà rencontrés dans le mélange.
    
    Paramètres:
        n (int): nombre de nœuds dans l'arbre.
        
    Retourne:
        tuple: (racine, arbre)
            - racine (int): le nœud choisi aléatoirement comme racine.
            - arbre (dict): dictionnaire représentant l'arbre orienté.
              Exemple: {3: [5, 0], 5: [8, 1], 0: [2, 4], 8: [], 1: [], 2: [], 4: []}
    """
    # Créer la liste des nœuds et la mélanger
    nodes = list(range(1, n+1))
    random.shuffle(nodes)
    
    # Le premier nœud de la liste mélangée est choisi comme racine
    racine = nodes[0]
    
    # Initialiser le dictionnaire pour chaque nœud avec une liste vide qui contiendra ses enfants
    arbre = {node: [] for node in nodes}
    
    # Pour chaque nœud (à partir du deuxième dans l'ordre mélangé),
    # choisir un parent aléatoire parmi les nœuds déjà placés avant lui dans la liste
    for i in range(1, len(nodes)):
        node = nodes[i]
        parent = random.choice(nodes[:i])
        arbre[parent].append(node)
    
    return racine, arbre

def creer_dag_aleatoire_connexe(n, p=0.3):
    """
    Crée un DAG aléatoire connexe (le graphe sous-jacent est connexe) représenté
    sous forme de dictionnaire (liste d'adjacence). La construction se fait en deux étapes :
    
      1. Création d'un arbre orienté couvrant (spanning tree) pour assurer la connexité.
         On choisit une racine aléatoire en mélangeant la liste des nœuds et, pour chaque nœud
         suivant, on connecte ce nœud à un nœud déjà rencontré.
         
      2. Ajout optionnel d'arêtes supplémentaires entre les paires de nœuds 
         respectant l'ordre établi (i < j) avec une probabilité p, sans dupliquer les arêtes existantes.
    
    Paramètres:
        n (int)   : nombre de nœuds du graphe.
        p (float) : probabilité d'ajouter une arête entre deux nœuds (en plus de l'arbre couvrant), par défaut 0.3.
        
    Retourne:
        tuple: (ordre, dag)
            - ordre (list) : liste des nœuds dans l'ordre aléatoire utilisé pour l'orientation du DAG.
            - dag (dict)   : dictionnaire représentant le DAG, où chaque clé est un nœud et
                             la valeur associée est la liste de ses successeurs.
    """
    # 1. Création et mélange des nœuds pour établir un ordre aléatoire
    nodes = list(range(n))
    random.shuffle(nodes)
    
    # Initialisation du DAG : chaque nœud aura une liste vide de successeurs
    dag = {node: [] for node in nodes}
    
    # 2. Création d'un arbre couvrant pour assurer la connexité
    # Le premier nœud de la liste mélangée est la racine
    for i in range(1, n):
        parent = random.choice(nodes[:i])
        dag[parent].append(nodes[i])
    
    # 3. Ajout d'arêtes supplémentaires en respectant l'ordre pour conserver l'acyclicité
    for i in range(n):
        for j in range(i+1, n):
            # On ajoute une arête de nodes[i] vers nodes[j] si elle n'existe pas déjà
            if nodes[j] not in dag[nodes[i]]:
                if random.random() < p:
                    dag[nodes[i]].append(nodes[j])
                    
    return nodes, dag

def create_file(filename, dag):
    """
    Crée un fichier texte contenant la représentation du DAG.
    
    Paramètres:
        filename (str) : nom du fichier à créer.
        dag (dict)     : dictionnaire représentant le DAG.
    """
    with open(filename, 'w') as f:
        t = len(dag)
        f.write(f"{t} {random.randint(1, t)}\n")
        for node, successors in dag.items():
            for successor in successors:
                f.write(f"{node} {successor}\n")

def create_n_dag(n, taille = 10, p = 0.3):
    """
    Crée n DAGs aléatoires et les enregistre dans un fichier texte.
    
    Paramètres:
        n (int)   : nombre de DAGs à créer.
        taille (int): nombre de nœuds dans chaque DAG.
        p (float) : probabilité d'ajouter une arête entre deux nœuds, par défaut 0.3.
    """
    dossier = Path("dag")
    # Crée le dossier s'il n'existe pas
    dossier.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        _, dag = creer_dag_aleatoire_connexe(taille, p)
        create_file(f"dag/dag_{i}.txt", dag)

def create_n_tree(n, taille = 10):
    """
    Crée n arbres orientés aléatoires et les enregistre dans un fichier texte.
    
    Paramètres:
        n (int)   : nombre d'arbres à créer.
        taille (int): nombre de nœuds dans chaque arbre.
    """
    dossier = Path("tree")
    # Crée le dossier s'il n'existe pas
    dossier.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        racine, arbre = creer_arbre_oriente_racine_aleatoire(taille)
        create_file(f"tree/arbre_{i}.txt", arbre)

# Exemple d'utilisation de la fonction
if __name__ == "__main__":
    n = 5   # Nombre de nœuds dans le DAG
    p = 0.3  # Probabilité pour l'ajout d'arêtes supplémentaires
    # create_n_dag(10, n, p)  # Crée 10 DAGs aléatoires avec 5 nœuds chacun
    create_n_tree(30, 10)  # Crée 10 arbres orientés aléatoires avec 5 nœuds chacun
    # create_n_dag(1, 100)
