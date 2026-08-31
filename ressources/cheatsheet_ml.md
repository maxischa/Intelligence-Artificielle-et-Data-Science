# Aide-mémoire machine learning — bloc 3

Gardez cette page ouverte pendant les exercices. Sur tablette, **copiez-collez**
depuis ici plutôt que de retaper.

La colonne **« ce qu'on en dit »** reste le vrai sujet : un modèle qui prédit
sans qu'on sache l'expliquer ne se déploie jamais.

---

## Les imports du bloc 3

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.metrics import precision_score, recall_score, silhouette_score
```

---

## Le squelette, toujours le même

```python
X = donnees[["colonne_a", "colonne_b"]]     # ce qu'on connait
y = donnees["cible"]                        # ce qu'on veut prevoir

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)
# random_state : n'importe quel nombre, du moment qu'il est FIXE

m = LinearRegression().fit(X_tr, y_tr)      # on APPREND sur X_tr
p = m.predict(X_te)                         # on PREDIT sur X_te
```

> **On apprend d'un côté, on note de l'autre.** Un modèle évalué sur les
> données qui l'ont produit ne mesure pas sa capacité à prédire, mais sa
> capacité à retenir.

---

## Prédire un nombre (3.1)

```python
mean_absolute_error(y_te, p)          # erreur moyenne, EN EUROS
mean_squared_error(y_te, p) ** 0.5    # RMSE : punit les grosses fautes
r2_score(y_te, p)                     # part expliquee, sans unite
```

| Ce que vous voyez | Ce qu'on en dit |
|---|---|
| excellent en apprentissage, mauvais en test | **surapprentissage** : brider le modèle |
| RMSE ≫ MAE | quelques prédictions sont franchement ratées, allez les voir |
| R² ≈ 1 | cherchez la **fuite de données** avant de vous réjouir |
| R² négatif | le modèle fait pire que « toujours la moyenne » |

> **Le test de la fuite :** pour chaque variable, *serait-elle disponible au
> moment où je dois décider ?* Si non, elle n'a rien à faire dans le modèle.

---

## Prédire une décision (3.1)

```python
X = pd.get_dummies(donnees.drop(columns=["cible"]), drop_first=True).astype(float)

m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
m.fit(X_tr, y_tr)

proba = m.predict_proba(X_te)[:, 1]   # la PROBABILITE, colonne 1
pred = (proba > 0.30).astype(int)     # le seuil est VOTRE decision
```

### Les quatre cases

|  | prédit : négatif | prédit : positif |
|---|---|---|
| **négatif vraiment** | vrai négatif (VN) | **faux positif** (FP) — une action pour rien |
| **positif vraiment** | **faux négatif** (FN) — un cas raté | vrai positif (VP) |

| Mesure | Formule | Ce qu'elle dit | Quand elle trompe |
|---|---|---|---|
| justesse | (VN + VP) / total | part de bonnes réponses | **toujours**, sur données déséquilibrées |
| précision | VP / (VP + FP) | parmi ceux qu'on cible, combien à raison | ignore ceux qu'on a ratés |
| rappel | VP / (VP + FN) | parmi les vrais cas, combien retrouvés | ignore les fausses alertes |
| F1 | 2 × préc. × rappel / (préc. + rappel) | les deux tiennent-elles ensemble | traite les deux erreurs comme si elles coûtaient pareil |

> 💡 Le F1 est une moyenne **harmonique** : elle est tirée vers le bas par la
> plus faible des deux. Appeler tout le monde donne un rappel de 1 et une
> précision de 0,27 — moyenne ordinaire 0,63, F1 seulement 0,42.

> ⚠️ **Toujours commencer par le modèle nul.** Prédire la classe majoritaire
> donne ici 73,4 % de justesse et zéro client sauvé.

**Choisir le seuil par le coût**, jamais par habitude :

```python
def gain(seuil, cout_contact, marge, taux_succes):
    p = (proba > seuil).astype(int)
    vrais = ((p == 1) & (y_te == 1)).sum()
    return vrais * taux_succes * marge - p.sum() * cout_contact
```

---

## Segmenter sans étiquette (3.2)

```python
# 1. Ecraser les extremes, puis egaliser les echelles. Les deux, dans cet
#    ordre : k-means mesure des DISTANCES, et une variable en euros ecraserait
#    une variable en jours.
Xs = StandardScaler().fit_transform(np.log1p(cli[["recence", "freq", "montant"]]))

# 2. Former k groupes
km = KMeans(n_clusters=4, n_init=10, random_state=42).fit(Xs)
cli["groupe"] = km.labels_

# 3. Les deux diagnostics
km.inertia_                       # baisse TOUJOURS quand k monte : lire le coude
silhouette_score(Xs, km.labels_)  # haute = groupes bien separes
```

| Ce que vous lisez | Ce qu'on en dit |
|---|---|
| un coude net | un nombre de groupes que les données suggèrent |
| silhouette entre 0,3 et 0,5 | des groupes qui existent mais se chevauchent — le cas normal |
| la silhouette préfère k = 2 | « les bons » et « les autres » : aucune action ne s'en déduit |

> **Le nombre de segments ne se lit pas dans une courbe.** Il se choisit sur le
> nombre de traitements commerciaux que l'entreprise peut mener de front. Le
> coude et la silhouette cadrent la décision, ils ne la prennent pas.

> ⚠️ **Identifiez les groupes par leur comportement, jamais par leur numéro.**
> `KMeans` ne les attribue pas dans un ordre stable : changez `random_state` et
> les numéros changent de place.

---

## L'atelier « The Inbox Problem » (3.3 et 3.4)

```python
%pip install -q sentence-transformers "google-genai==2.9.0"

from sentence_transformers import SentenceTransformer
encodeur = SentenceTransformer("all-MiniLM-L6-v2")   # 91 Mo, gratuit, sans cle
E = encodeur.encode(messages)                        # (n, 384) : le SENS en position

# La cle vit dans les Secrets de Colab, JAMAIS dans une cellule
from google import genai
from google.colab import userdata
ai = genai.Client(api_key=userdata.get("GEMINI_API_KEY"))
reponse = ai.interactions.create(model="gemini-3.5-flash-lite", input=consigne)
print(reponse.output_text)
```

| Le mot | Ce qu'il veut dire |
|---|---|
| **embedding** | une carte où le sens est une position |
| **similarité cosinus** | la distance entre deux points sur cette carte |
| **seuil de confiance** | à quel point le modèle doit être sûr pour décider seul |
| **baseline** | la politique la plus bête. Si le modèle ne la bat pas, il n'y a pas de modèle |
| **hallucination** | combler un trou par du plausible plutôt que dire « je ne sais pas » |

---

## Les erreurs les plus fréquentes du bloc 3

| Message | Cause | Solution |
|---|---|---|
| `could not convert string to float: 'mensuel'` | colonnes de texte | `pd.get_dummies(...)` |
| `Found input variables with inconsistent numbers of samples` | jeux mélangés | prédire et évaluer sur le **même** jeu |
| `NotFittedError` | `.fit()` oublié | `Modele().fit(X_tr, y_tr)` |
| `MemoryError` / la session redémarre | trop de messages encodés d'un coup | réduire `batch_size`, ou encoder par tranches |
| un R² de 1 **sans erreur** | fuite de données | une variable contient la réponse |
| 73 % de justesse **sans erreur** | classes déséquilibrées | comparez au modèle nul, regardez le rappel |
| une catégorie inventée avec assurance **sans erreur** | le LLM ne dit jamais « je ne sais pas » | prévoyez une valeur de repli et relisez |
| un `KeyError` sur `GEMINI_API_KEY` | secret absent ou notebook non autorisé | panneau **Secrets**, nom exact, accès coché |

> **Ne lisez que la dernière ligne d'un message d'erreur.** Et méfiez-vous
> surtout des trois lignes de ce tableau qui n'en produisent aucun.
