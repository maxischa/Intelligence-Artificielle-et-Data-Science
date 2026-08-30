"""Construit les jeux de donnees du bloc 3 (Machine learning).

Deux fichiers, deux origines.

    commandes.csv   une commande du detaillant par ligne   (1 955 lignes)
    churn.csv       un abonne telecom par ligne            (7 043 lignes)

`commandes.csv` derive des CSV du bloc 2 : on reste chez le meme detaillant,
mais a une MAILLE differente. Le bloc 2 raisonne par ligne de vente ; decrire
et predire demandent des individus statistiques comparables entre eux. C'est la
table de la seance 3.1 (echauffement statistique) et de la seance 3.2 (predire
un montant).

`churn.csv` vient du jeu public IBM Telco Customer Churn. C'est l'etude de cas
des seances 3.3 et 3.4 : une cible binaire nette (26,5 % de resiliations), des
variables melangeant quantitatif et qualitatif, et surtout des facteurs
dominants spectaculaires — le contrat mensuel resilie a 42,7 %, le contrat de
deux ans a 2,8 %.

> Les colonnes de `churn.csv` sont renommees en ASCII court (`tenure` -> `anc`,
> `MonthlyCharges` -> `mensuel`), et les modalites traduites. Ce n'est pas de
> la coquetterie : chaque filtre tape sur tablette coute sinon trente frappes,
> et `Month-to-month` en contient quatorze a lui seul.

> Attention : `segment` n'est PAS repris dans `commandes.csv`. Dans
> `clients.csv` du bloc 2, il est construit par `pd.qcut` sur le CA total du
> client. Le tester ou le regresser contre `ca` serait tautologique, et
> donnerait des p-values ecrasantes qui n'apprennent rien. Les variables
> qualitatives de ce bloc sont `pays` et `jour`.

Ce script reunit ceux des anciens blocs 3 et 4 du cours 36h, dont ce bloc-ci
est la fusion. Trois tables ont disparu avec les seances qu'elles servaient :
`clients_ca.csv` (regression sur statsmodels), `clients_rfm.csv` et
`produits_profil.csv` (segmentation k-means).

Ce script n'est PAS execute par les etudiants : les CSV sont commites dans le
depot et lus par URL. Il est versionne pour que la construction reste
reproductible.

Usage :
    python bloc3_ml/data/build_data.py
"""

import urllib.request
from pathlib import Path

import pandas as pd

ICI = Path(__file__).parent
RETAIL = ICI.parent.parent / "bloc2_donnees" / "data"

URL_CHURN = ("https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
             "master/data/Telco-Customer-Churn.csv")

# Les jours sont tapes dans des filtres (jour == "jeudi") : ASCII minuscule.
JOURS = {
    "Monday": "lundi",
    "Tuesday": "mardi",
    "Wednesday": "mercredi",
    "Thursday": "jeudi",
    "Friday": "vendredi",
    "Saturday": "samedi",
    "Sunday": "dimanche",
}

# Colonnes retenues sur les 21 d'origine. Les 19 explicatives du fichier source
# debordent d'un ecran de tablette et diluent l'interpretation : on garde
# celles qui portent le signal, et une seule variable de service (internet).
COLONNES = {
    "tenure": "anc",              # anciennete en mois
    "MonthlyCharges": "mensuel",  # facture mensuelle
    "TotalCharges": "total",      # facture cumulee
    "Contract": "contrat",
    "InternetService": "internet",
    "PaymentMethod": "paiement",
    "SeniorCitizen": "senior",
    "Partner": "couple",
    "TechSupport": "support",
    "Churn": "churn",
}

MODALITES = {
    "contrat": {"Month-to-month": "mensuel", "One year": "un_an", "Two year": "deux_ans"},
    "internet": {"Fiber optic": "fibre", "DSL": "dsl", "No": "aucun"},
    "paiement": {"Electronic check": "cheque_el", "Mailed check": "cheque",
                 "Bank transfer (automatic)": "virement",
                 "Credit card (automatic)": "carte"},
    "couple": {"Yes": "oui", "No": "non"},
    "support": {"Yes": "oui", "No": "non", "No internet service": "aucun"},
    "churn": {"Yes": 1, "No": 0},
}


def construire_commandes() -> pd.DataFrame:
    """Un panier par ligne, agrege depuis les ventes du bloc 2.

    Pourquoi pre-agreger plutot que leur faire refaire les merge du bloc 2 :
    sur tablette, c'est dix minutes de re-frappe au debut de chaque seance,
    pour un resultat deja valide. Le temps gagne va a l'interpretation.

    `nart` (nombre de references) et `qte` (nombre d'articles) sont gardes tous
    les deux : leur correlation au CA est tres differente, et la comparaison
    est exactement le sujet du §2 de la seance 3.1.
    """
    ventes = pd.read_csv(RETAIL / "ventes.csv", parse_dates=["date"])
    clients = pd.read_csv(RETAIL / "clients.csv", parse_dates=["date_insc"])

    ventes["ca"] = ventes["qte"] * ventes["prix"]
    vc = ventes.merge(clients[["client_id", "pays"]], on="client_id")
    assert len(vc) == len(ventes), "le merge a duplique ou perdu des lignes"

    commandes = (
        vc.groupby("cmd_id", as_index=False)
        .agg(
            date=("date", "first"),
            ca=("ca", "sum"),
            nart=("prod_id", "count"),
            qte=("qte", "sum"),
            pays=("pays", "first"),
            client_id=("client_id", "first"),
        )
    )
    commandes["jour"] = commandes["date"].dt.day_name().map(JOURS)
    commandes["ca"] = commandes["ca"].round(2)
    commandes["date"] = commandes["date"].dt.strftime("%Y-%m-%d %H:%M")
    commandes = commandes[
        ["cmd_id", "date", "jour", "ca", "nart", "qte", "pays", "client_id"]
    ].sort_values("date").reset_index(drop=True)

    # Le samedi est absent du fichier source : l'enseigne ne traite aucune
    # commande ce jour-la. C'est une asperite de la seance 3.1 — un zero qui
    # n'apparait nulle part. Si un jour la source change, l'exercice tombe :
    # d'ou ce controle.
    assert "samedi" not in set(commandes["jour"]), "le samedi n'est plus vide"
    return commandes


def construire_churn() -> pd.DataFrame:
    """Telecharge le jeu IBM, reduit les colonnes, traduit les modalites."""
    cache = ICI / "_source_telco.csv"
    if not cache.exists():
        print("Telechargement du jeu IBM Telco Customer Churn...")
        urllib.request.urlretrieve(URL_CHURN, cache)
    brut = pd.read_csv(cache)

    churn = brut[list(COLONNES)].rename(columns=COLONNES)
    for colonne, table in MODALITES.items():
        churn[colonne] = churn[colonne].map(table)

    # `total` est laisse EN TEXTE, avec ses onze cases vides : c'est le premier
    # geste de la seance 3.3, et un rappel direct du nettoyage de la 2.1.
    manquants = pd.to_numeric(churn["total"], errors="coerce").isna().sum()
    print(f"  {manquants} valeurs de `total` non convertibles, conservees telles quelles")

    assert churn["churn"].notna().all(), "une modalite de churn n'a pas ete traduite"
    return churn


def main() -> None:
    commandes = construire_commandes()
    churn = construire_churn()

    for nom, table in [("commandes.csv", commandes), ("churn.csv", churn)]:
        chemin = ICI / nom
        table.to_csv(chemin, index=False)
        taille = chemin.stat().st_size / 1e6
        print(f"{nom:<16} {len(table):>6} lignes  {taille:5.2f} Mo")

    print(f"jours presents : {commandes['jour'].nunique()} (samedi absent, comme attendu)")
    print(f"taux de resiliation : {100 * churn['churn'].mean():.1f} %")


if __name__ == "__main__":
    main()
