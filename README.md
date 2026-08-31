# Intelligence Artificielle et Data Science

Introduction à Python pour la collecte et l'analyse de données — 14 heures.

> 📱 **Vous travaillez sur tablette ?** Lisez d'abord
> **[Bien démarrer](ressources/setup_tablette.md)**. Cinq minutes de réglages
> vous éviteront la plupart des blocages.

> ⚠️ À l'ouverture de chaque notebook : **Fichier → Enregistrer une copie dans
> Drive**, *avant* de taper quoi que ce soit. Sinon votre travail est perdu à
> la fermeture de l'onglet.

---

## Bloc 1 — Prise en main de Python et de Colab (2h)

Une séance de 2h. Le notebook de **cours** est suivi en séance, les **exercices** se font en autonomie, la **correction** est publiée après.

| Séance | Sujet | Cours | Exercices | Correction |
|---|---|---|---|---|
| 1.1 | Prise en main — Colab, Markdown et vos premières lignes | [▶](https://colab.research.google.com/github/maxischa/Intelligence-Artificielle-et-Data-Science/blob/main/bloc1_python/cours/seance1_cours.ipynb) | [▶](https://colab.research.google.com/github/maxischa/Intelligence-Artificielle-et-Data-Science/blob/main/bloc1_python/exercices/seance1_exercices.ipynb) | [▶](https://colab.research.google.com/github/maxischa/Intelligence-Artificielle-et-Data-Science/blob/main/bloc1_python/corrections/seance1_correction.ipynb) |

📄 **[Bien démarrer — surtout sur tablette](ressources/setup_tablette.md)** — à garder ouvert pendant les exercices.

---

### Les données du bloc 1

Un seul fichier, minuscule, pour la dernière cellule de la séance :
charger des données depuis le web tient en une ligne, et c'est tout
le bloc 2 qui commence là.

| Fichier | Lignes | Contenu |
|---|---|---|
| `premieres_ventes.csv` | 20 | Vingt lignes du détaillant du bloc 2 : `date`, `produit`, `qte`, `prix`, `pays` |

Extrait de `bloc2_donnees/data/ventes.csv`. Construction reproductible par [`bloc1_python/data/build_data.py`](bloc1_python/data/build_data.py).

---

## Bloc 2 — Collecter, comprendre et manipuler des données (4h)

2 séances de 2h. Pour chacune : le notebook de **cours** est suivi en séance, les **exercices** se font en autonomie, la **correction** est publiée après.

| Séance | Sujet | Cours | Exercices | Correction |
|---|---|---|---|---|
| 2.1 | Charger, comprendre et nettoyer un jeu de données | [▶](https://colab.research.google.com/github/maxischa/Intelligence-Artificielle-et-Data-Science/blob/main/bloc2_donnees/cours/seance1_cours.ipynb) | [▶](https://colab.research.google.com/github/maxischa/Intelligence-Artificielle-et-Data-Science/blob/main/bloc2_donnees/exercices/seance1_exercices.ipynb) | [▶](https://colab.research.google.com/github/maxischa/Intelligence-Artificielle-et-Data-Science/blob/main/bloc2_donnees/corrections/seance1_correction.ipynb) |
| 2.2 | Agréger, croiser et visualiser — étude de cas | [▶](https://colab.research.google.com/github/maxischa/Intelligence-Artificielle-et-Data-Science/blob/main/bloc2_donnees/cours/seance2_cours.ipynb) | [▶](https://colab.research.google.com/github/maxischa/Intelligence-Artificielle-et-Data-Science/blob/main/bloc2_donnees/exercices/seance2_exercices.ipynb) | [▶](https://colab.research.google.com/github/maxischa/Intelligence-Artificielle-et-Data-Science/blob/main/bloc2_donnees/corrections/seance2_correction.ipynb) |

📄 **[Aide-mémoire pandas](ressources/cheatsheet_pandas.md)** — à garder ouvert pendant les exercices.

---

### Les données du bloc 2

Un détaillant en ligne européen, décembre 2010 à décembre 2011.
Les fichiers se chargent **directement depuis le web** : rien à télécharger.

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

4 séances de 2h. Les **deux premières** suivent le format habituel — cours en séance, exercices en autonomie, correction publiée après. Les **deux dernières** forment l'atelier *The Inbox Problem* : une **démonstration** de 4h, que l'enseignant déroule et que vous suivez sur votre copie. Elles n'ont donc ni feuille d'exercices ni correction, et demandent une **clé API Gemini** gratuite.

| Séance | Sujet | Cours | Exercices | Correction |
|---|---|---|---|---|
| 3.1 | Prédire un nombre, prédire une décision | [▶](https://colab.research.google.com/github/maxischa/Intelligence-Artificielle-et-Data-Science/blob/main/bloc3_ml/cours/seance1_cours.ipynb) | [▶](https://colab.research.google.com/github/maxischa/Intelligence-Artificielle-et-Data-Science/blob/main/bloc3_ml/exercices/seance1_exercices.ipynb) | [▶](https://colab.research.google.com/github/maxischa/Intelligence-Artificielle-et-Data-Science/blob/main/bloc3_ml/corrections/seance1_correction.ipynb) |
| 3.2 | Segmenter sans étiquette — quatre clients, quatre traitements | [▶](https://colab.research.google.com/github/maxischa/Intelligence-Artificielle-et-Data-Science/blob/main/bloc3_ml/cours/seance2_cours.ipynb) | [▶](https://colab.research.google.com/github/maxischa/Intelligence-Artificielle-et-Data-Science/blob/main/bloc3_ml/exercices/seance2_exercices.ipynb) | [▶](https://colab.research.google.com/github/maxischa/Intelligence-Artificielle-et-Data-Science/blob/main/bloc3_ml/corrections/seance2_correction.ipynb) |
| 3.3 | The Inbox Problem (1/2) — du mot au sens | [▶](https://colab.research.google.com/github/maxischa/Intelligence-Artificielle-et-Data-Science/blob/main/bloc3_ml/cours/seance3_cours.ipynb) | — | — |
| 3.4 | The Inbox Problem (2/2) — de la carte à la décision | [▶](https://colab.research.google.com/github/maxischa/Intelligence-Artificielle-et-Data-Science/blob/main/bloc3_ml/cours/seance4_cours.ipynb) | — | — |

📄 **[Aide-mémoire machine learning](ressources/cheatsheet_ml.md)** · **[Aide-mémoire pandas](ressources/cheatsheet_pandas.md)** — à garder ouvert pendant les exercices.

---

### Les données du bloc 3

Trois terrains. Le détaillant des blocs 1 et 2 revient à la maille de
la commande ; un opérateur télécom fournit une cible binaire nette ;
et 5 000 messages réels de clients d'une banque en ligne servent
l'atelier final. Ces derniers sont en **anglais** : c'est un corpus
réel et étiqueté, et il n'en existe pas d'équivalent public en
français.

| Fichier | Lignes | Contenu |
|---|---|---|
| `commandes.csv` | 1 955 | Une commande par ligne : `cmd_id`, `date`, `jour`, `ca`, `nart`, `qte`, `pays`, `client_id` |
| `churn.csv` | 7 043 | Un abonné télécom par ligne : `anc`, `mensuel`, `total`, `contrat`, `internet`, `paiement`, `senior`, `couple`, `support`, `churn` |
| `clients_rfm.csv` | 472 | Un client par ligne : `recence`, `freq`, `montant`, `pays` |
| `produits_profil.csv` | 1 263 | Une référence vendue au moins 10 fois : `nb_cmd`, `qte`, `ca`, `prix`, `pays`, `clients`, `part_q4` |
| `tickets.csv` | 5 000 | Un message client par ligne : `message`, `intention`, `equipe` |
| `tickets_nouveaux.csv` | 500 | Le lot inédit du défi final, mêmes colonnes |

`commandes.csv`, `clients_rfm.csv` et `produits_profil.csv` dérivent des fichiers du bloc 2.
`churn.csv` vient d'[IBM Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d).
Les tickets viennent de [banking77](https://github.com/PolyAI-LDN/task-specific-datasets) (PolyAI, CC-BY), dont les 77 intentions sont regroupées en 8 équipes de routage.
Construction reproductible par [`bloc3_ml/data/build_data.py`](bloc3_ml/data/build_data.py).

---

## Pour l'équipe enseignante

Ce dépôt est **purement étudiant**. La chaîne de production du cours reste sur
la machine de l'enseignant et n'est pas publiée :

| Reste local | Pourquoi |
|---|---|
| `bloc*/intervenant/` | cours minuté, notes de passation, pièges attendus |
| `outils/` | la source dont les notebooks sont générés — contient les mêmes notes et toutes les solutions |

Les quatre variantes d'une séance (cours, intervenant, exercices, correction)
sont **générées depuis une source unique**, une variante par usage. Un énoncé
ne peut donc pas diverger entre le notebook d'exercices et sa correction, et
une solution ne peut pas se retrouver par accident dans le notebook remis aux
étudiants.

Côté enseignant, trois commandes :

```bash
python outils/construire_notebooks.py   # regenere les notebooks
python outils/construire_readme.py      # regenere cette page
python outils/verifier_notebooks.py     # les execute tous et controle les regles
```

L'adresse de ce dépôt est définie à un seul endroit (`outils/depot.py`) :
la changer et regénérer suffit à mettre à jour tous les badges « Open in
Colab », tous les liens de cette page et toutes les URL de données.
