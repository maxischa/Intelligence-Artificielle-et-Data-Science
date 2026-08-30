# Aide-mémoire statistiques — séance 3.1

Gardez cette page ouverte pendant les exercices. Sur tablette, **copiez-collez**
depuis ici plutôt que de retaper.

L'aide-mémoire pandas reste valable : on continue de charger, filtrer et
grouper. Ce qui change, c'est ce qu'on fait ensuite. La colonne **« ce qu'on
en dit »** est le vrai sujet — une commande sans interprétation ne vaut rien.

Dans tous les exemples, `cmd` désigne la table des commandes.

---

## Les imports

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats        # pour le test de comparaison
```

---

## 1. Décrire une distribution

```python
cmd["ca"].describe()          # count, mean, std, min, quartiles, max
cmd["ca"].mean()              # la moyenne
cmd["ca"].median()            # la mediane
cmd["ca"].std()               # l'ecart-type
cmd["ca"].quantile(0.9)       # le seuil des 10 % du haut
cmd["ca"].nunique()           # nombre de valeurs distinctes

q1, q3 = cmd["ca"].quantile(0.25), cmd["ca"].quantile(0.75)
q3 - q1                       # l'ecart interquartile, version robuste

cmd["ca"].plot(kind="hist", bins=40, figsize=(7, 4))   # la FORME
```

| Ce que vous lisez | Ce qu'on en dit |
|---|---|
| moyenne ≫ médiane | distribution asymétrique : **citez la médiane** |
| écart-type > moyenne | **indicateur** de dispersion très forte : ne citez pas la moyenne seule |
| un groupe à faible effectif | ne commentez pas : regardez `count` d'abord |

```python
# La concentration : quelle part du total tient dans le haut du classement
top = cmd["ca"].sort_values(ascending=False)
100 * top.head(int(0.10 * len(cmd))).sum() / top.sum()
```

```python
# Decouper une quantitative en tranches : les bornes sont un CHOIX,
# annoncez-les avec vos resultats. 5 bornes -> 4 tranches.
cmd["gamme"] = pd.cut(cmd["ca"], bins=[0, 200, 500, 1000, 20000],
                      labels=["petite", "moyenne", "grande", "tres_grande"])
```

---

## 2. Relier deux variables

**On trace d'abord. Toujours.**

```python
cmd.plot(kind="scatter", x="qte", y="ca", alpha=0.3, figsize=(7, 4))

cmd[["ca", "nart", "qte"]].corr()                    # Pearson (droite)
cmd["ca"].corr(cmd["nart"], method="spearman")       # Spearman (rangs)
```

| Ce que vous voyez | Ce qu'on en dit |
|---|---|
| Pearson ≪ Spearman | la relation n'est pas droite, ou les extrêmes pèsent |
| corrélation ≈ 0 | aucun lien **de forme droite** — regardez le nuage |
| corrélation très forte | vérifiez que l'une ne sert pas à calculer l'autre |

Avant de commenter un coefficient, deux questions :

1. **Les deux variables sont-elles mesurées indépendamment ?** `ca` et `qte`
   corrèlent à 0,85 parce que le montant se calcule à partir des quantités.
   Ça ne dit rien du comportement d'achat.
2. **Le coefficient décrit-il une seule population ?** 0,38 en global, 0,90 en
   Belgique, 0,23 en Irlande. Cherchez le **Z** qui expliquerait à la fois X
   et Y — ici, « ce client est un grossiste ».

---

## 3. Comparer deux groupes

```python
fr = cmd.query("pays == 'France'")["ca"]
de = cmd.query("pays == 'Allemagne'")["ca"]

# equal_var=False : on ne suppose pas la meme dispersion des deux cotes
stats.ttest_ind(fr, de, equal_var=False).pvalue
```

| p-value | Ce qu'on en dit |
|---|---|
| **p < 0,05** | l'écart serait rare si les groupes étaient identiques : on le retient |
| **p ≥ 0,05** | l'écart est compatible avec le hasard : **on ne conclut rien** |

> « Pas de différence détectable » **n'est pas** « pas de différence ».

Avant tout test, deux vérifications :

```python
cmd.groupby("pays")["client_id"].nunique()   # combien d'individus, pas de lignes
round(irl.mean() - uk.mean(), 2)             # la taille de l'effet, en euros
```

> **Significatif ne veut pas dire important.** Sur 100 000 observations, un
> écart de 3 € sort avec p < 0,001. Affichez toujours l'écart dans son unité
> à côté de la p-value.

---

## Pour la partie 2 de la feuille

Ces commandes ne sont pas dans le notebook de cours : les énoncés de la
partie 2 les introduisent au fil de l'eau.

```python
df.groupby("pays")["ca"].describe()      # un describe() par groupe
df.copy()                                # avant de modifier, toujours
df["ca"].idxmax()                        # l'etiquette de la plus grande valeur
df["jour"].unique()                      # les modalites REELLEMENT presentes
np.log(serie)                            # redresse une relation courbe
df.plot(kind="scatter", x="a", y="b", loglog=True)   # les deux axes en log
df.groupby(["pays", "client_id"])["ca"].mean().reset_index()   # au bon niveau
```

> ⚠️ Une corrélation calculée sur des données **agrégées** est toujours plus
> forte : par commande, par client, par pays — mêmes données, trois chiffres.
> Précisez toujours le niveau d'observation.

---

## Les erreurs les plus fréquentes

| Message | Cause | Solution |
|---|---|---|
| `ValueError: percentiles should all be in the interval [0, 1]` | `quantile(90)` au lieu de `quantile(0.9)` | un quantile est une **proportion** |
| `ValueError: Bin labels must be one fewer than the number of bin edges` | `pd.cut` : autant d'étiquettes que de bornes | 5 bornes → 4 étiquettes |
| `TypeError: Could not convert string ... to numeric` | moyenne sur du texte | vérifiez la colonne |
| `pvalue = nan` **sans erreur** | un des deux groupes est vide | vérifiez l'orthographe du filtre |
| une corrélation nulle **sans erreur** | relation non linéaire | tracez le nuage de points |
| une moyenne « typique » trompeuse **sans erreur** | distribution asymétrique | affichez aussi la médiane |

> **Ne lisez que la dernière ligne d'un message d'erreur.** Et méfiez-vous
> surtout des trois dernières lignes de ce tableau, qui n'en produisent aucun.
