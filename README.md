# Intelligence Artificielle et Data Science

> 📱 **Vous travaillez sur tablette ?** Lisez d'abord
> **[Bien démarrer](ressources/setup_tablette.md)**. Cinq minutes de réglages
> vous éviteront la plupart des blocages.

> ⚠️ À l'ouverture de chaque notebook : **Fichier → Enregistrer une copie dans
> Drive**, *avant* de taper quoi que ce soit. Sinon votre travail est perdu à
> la fermeture de l'onglet.

> 🗓️ **Le dépôt se remplit au fil des séances.** Le cours et les exercices
> paraissent avant le créneau, la correction après. Les lignes marquées
> *à venir* ne sont pas des oublis : elles s'ouvriront le moment venu.

---

## Bloc 1 — Prise en main de Python et de Colab (2h)

| Séance | Sujet | Cours | Exercices | Correction |
|---|---|---|---|---|
| 1.1 | Prise en main — Colab, Markdown et vos premières lignes | [▶](https://colab.research.google.com/github/maxischa/Intelligence-Artificielle-et-Data-Science/blob/main/bloc1_python/cours/seance1_cours.ipynb) | [▶](https://colab.research.google.com/github/maxischa/Intelligence-Artificielle-et-Data-Science/blob/main/bloc1_python/exercices/seance1_exercices.ipynb) | [▶](https://colab.research.google.com/github/maxischa/Intelligence-Artificielle-et-Data-Science/blob/main/bloc1_python/corrections/seance1_correction.ipynb) |

📄 **[Bien démarrer — surtout sur tablette](ressources/setup_tablette.md)** — à garder ouvert pendant les exercices.

---

### Les données du bloc 1

| Fichier | Lignes | Contenu |
|---|---|---|
| `premieres_ventes.csv` | 20 | Vingt lignes du détaillant du bloc 2 : `date`, `produit`, `qte`, `prix`, `pays` |

Extrait de `bloc2_donnees/data/ventes.csv`. Construction reproductible par [`bloc1_python/data/build_data.py`](bloc1_python/data/build_data.py).

---

## Bloc 2 — Collecter, comprendre et manipuler des données (4h)

| Séance | Sujet | Cours | Exercices | Correction |
|---|---|---|---|---|
| 2.1 | Charger, comprendre et nettoyer un jeu de données | [▶](https://colab.research.google.com/github/maxischa/Intelligence-Artificielle-et-Data-Science/blob/main/bloc2_donnees/cours/seance1_cours.ipynb) | [▶](https://colab.research.google.com/github/maxischa/Intelligence-Artificielle-et-Data-Science/blob/main/bloc2_donnees/exercices/seance1_exercices.ipynb) | [▶](https://colab.research.google.com/github/maxischa/Intelligence-Artificielle-et-Data-Science/blob/main/bloc2_donnees/corrections/seance1_correction.ipynb) |
| 2.2 | Agréger, croiser et visualiser | [▶](https://colab.research.google.com/github/maxischa/Intelligence-Artificielle-et-Data-Science/blob/main/bloc2_donnees/cours/seance2_cours.ipynb) | [▶](https://colab.research.google.com/github/maxischa/Intelligence-Artificielle-et-Data-Science/blob/main/bloc2_donnees/exercices/seance2_exercices.ipynb) | *à venir* |

📄 **[Aide-mémoire pandas](ressources/cheatsheet_pandas.md)** — à garder ouvert pendant les exercices.

---

### Les données du bloc 2

| Fichier | Lignes | Contenu |
|---|---|---|
| `ventes.csv` | 45 123 | Une ligne par produit vendu : `date`, `cmd_id`, `prod_id`, `qte`, `prix`, `client_id` |
| `clients.csv` | 472 | Un client par ligne : `client_id`, `pays`, `segment`, `date_insc` |
| `produits.csv` | 2 956 | Un produit par ligne : `prod_id`, `libelle`, `categorie` |
| `ventes_sale.csv` | 5 370 | Un extrait **volontairement sale**, pour le nettoyage |

Source : [UCI Machine Learning Repository — Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii).
Construction reproductible par [`bloc2_donnees/data/build_data.py`](bloc2_donnees/data/build_data.py).

---

## Bloc 3 — Machine Learning and AI in Data Science (8h)

| Séance | Sujet | Cours | Exercices | Correction |
|---|---|---|---|---|
| 3.1 | Prédire un nombre, prédire une décision | *à venir* | *à venir* | *à venir* |
| 3.2 | Segmenter sans étiquette — quatre clients, quatre traitements | *à venir* | *à venir* | *à venir* |
| 3.3 | The Inbox Problem (1/2) — du mot au sens | *à venir* | — | — |
| 3.4 | The Inbox Problem (2/2) — de la carte à la décision | *à venir* | — | — |

📄 **[Aide-mémoire machine learning](ressources/cheatsheet_ml.md)** · **[Aide-mémoire pandas](ressources/cheatsheet_pandas.md)** — à garder ouvert pendant les exercices.

---

### Les données du bloc 3

| Fichier | Lignes | Contenu |
|---|---|---|
| `commandes.csv` | 1 955 | Une commande par ligne : `cmd_id`, `date`, `jour`, `ca`, `nart`, `qte`, `pays`, `client_id` |
| `churn.csv` | 7 043 | Un abonné télécom par ligne : `anc`, `mensuel`, `total`, `contrat`, `internet`, `paiement`, `senior`, `couple`, `support`, `churn` |
| `clients_rfm.csv` | 472 | Un client par ligne : `recence`, `freq`, `montant`, `pays` |
| `produits_profil.csv` | 1 263 | Une référence vendue au moins 10 fois : `nb_cmd`, `qte`, `ca`, `prix`, `pays`, `clients`, `part_q4` |
| `tickets.csv` | 5 000 | Un message client par ligne : `message`, `intention`, `equipe` |

`commandes.csv`, `clients_rfm.csv` et `produits_profil.csv` dérivent des fichiers du bloc 2.
`churn.csv` vient d'[IBM Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d).
Les tickets viennent de [banking77](https://github.com/PolyAI-LDN/task-specific-datasets) (PolyAI, CC-BY), dont les 77 intentions sont regroupées en 8 équipes de routage.
Construction reproductible par [`bloc3_ml/data/build_data.py`](bloc3_ml/data/build_data.py).
