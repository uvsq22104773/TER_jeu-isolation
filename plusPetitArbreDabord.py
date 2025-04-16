from collections import defaultdict

# Définition des arêtes de l'arbre
edges = [
    (0,1), (0,2), (0,3), (1,4), (2,5), (2,6), (2,7),
    (3,8), (3,9), (6,10), (9,11), (9,12), (8, 13)
]

# Construction de l'arbre : dictionnaire des enfants
tree = defaultdict(list)
for parent, child in edges:
    tree[parent].append(child)

# Fonction pour calculer la taille du sous-arbre (récursif)
def calculer_sous_arbre(sommet):
    if sommet not in tree:
        return 1  # Feuille
    return 1 + sum(calculer_sous_arbre(fils) for fils in tree[sommet])

# L'algo principal (traduction du pseudo-code)
def algo(ls, res):
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
            nb = calculer_sous_arbre(fs)
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
        sub_res, sub_total = algo(aTraiter, res)
        return res, maxGlobal + sub_total

# Appel de l'algo sur le sommet racine [0]
result, total = algo([0], [])

# Affichage
print("Arêtes supprimées : ", result)
print("Nombre de sommets sauvé : ", total)
