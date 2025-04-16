from collections import defaultdict

# Définition des arêtes de l'arbre
edges = [
    (0,1), (0,2), (0,3), (1,4), (2,5), (2,6), (2,7),
    (3,8), (3,9), (6,10), (9,11), (9,12), (8,13)
]

# Construction de l'arbre
tree = defaultdict(list)
for parent, child in edges:
    tree[parent].append(child)

# Fonction de calcul récursif du nombre de sommets dans un sous-arbre
def calcul(sommet):
    if sommet not in tree:
        return 1
    return 1 + sum(calcul(fils) for fils in tree[sommet])

# Algo2 complet
def algo2(ls, res):
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
            nbSommet = calcul(fs)

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
        sub_res, sub_total = algo2(aTraiter, res)
        return res, maxGlobal + sub_total

# Lancement de l'algo
result, total = algo2([0], [])

# Affichage
print("Arêtes supprimées :", result)
print("Nombre de sommets sauvés :", total)
