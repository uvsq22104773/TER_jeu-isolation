from dataclasses import dataclass

# Structure pour un graph
@dataclass
class Graph:
    t : int
    A : list

# Fonction qui charge un graph 
# contenue dans un fichier texte
def lecture_fichier(file: str) -> Graph:
    # Test d'ouverture du fichier 
    # et gestion des erreurs
    try:
        F = open(file, 'r', encoding='utf-8')
    except FileNotFoundError:
        print(f"Erreur : Le fichier '{file}' n'existe pas.")
        return None
    except IOError as e:
        print(f"Erreur lors de l'ouverture du fichier : {e}")
        return None
    # Après ouverture créer l'objet graph 
    # et lui assigne la matrice du fichier
    ligne = F.readline()
    taille = int(ligne)
    graph = Graph(
        t = taille,
        A = [[0 for _ in range(taille)] for _ in range(taille)]
    )
    i = 0
    ligne = F.readline()
    # Assigne chaque du fichier une à 
    # une dans la matrice du graph
    while ligne:
        valeurs = ligne.split()
        j = 0
        for valeur in valeurs:
            graph.A[i][j] = int(valeur)
            j += 1
        ligne = F.readline()
        i += 1
    # Fermeture du fichier
    F.close()
    # Retourne le graph
    return graph

G = lecture_fichier('graph1.txt')
