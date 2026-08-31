"""Construit les jeux de donnees du bloc 3 (Machine Learning and AI in Data Science).

Six fichiers, trois origines.

    commandes.csv        une commande du detaillant par ligne   (1 955 lignes)
    churn.csv            un abonne telecom par ligne            (7 043 lignes)
    clients_rfm.csv      un client du detaillant                (  472 lignes)
    produits_profil.csv  une reference vendue >= 10 fois        (1 263 lignes)
    tickets.csv          un message client par ligne            (5 000 lignes)
    tickets_nouveaux.csv le lot inedit du defi final            (  500 lignes)

`commandes.csv` derive des CSV du bloc 2 : on reste chez le meme detaillant,
mais a une MAILLE differente. Le bloc 2 raisonne par ligne de vente ; decrire
et predire demandent des individus statistiques comparables entre eux. C'est la
table de la seance 3.1 (echauffement statistique) et de la seance 3.2 (predire
un montant).

`clients_rfm.csv` et `produits_profil.csv` derivent eux aussi du bloc 2 : ce
sont les tables de la seance 3.2 (segmentation).

`tickets.csv` vient de banking77 (PolyAI, CC-BY) : 13 000 messages reels de
clients d'une banque en ligne, etiquetes par intention. Les 77 intentions
d'origine sont regroupees en HUIT equipes de routage — ce regroupement n'est
pas un detail technique, c'est la premiere decision de gestion de l'atelier
« The Inbox Problem » (seances 3.3 et 3.4). Le texte reste en anglais : c'est
un corpus reel et etiquete, et il n'en existe pas d'equivalent public en
francais. Meme situation qu'au bloc 2, dont les libelles produits sont anglais.

`churn.csv` vient du jeu public IBM Telco Customer Churn. C'est l'etude de cas
de la seance 3.1 : une cible binaire nette (26,5 % de resiliations), des
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
est la fusion, et y ajoute le corpus de tickets. Une seule table du cours
d'origine a disparu : `clients_ca.csv`, qui servait la regression sur
statsmodels, supprimee.

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

URL_BANKING = ("https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/"
               "master/banking_data/")

# Les 77 intentions de banking77, regroupees en HUIT equipes de routage.
# Ce regroupement est une decision de GESTION, pas un pretraitement : c'est
# l'organigramme du service client qu'on decrit ici, et l'atelier le discute
# explicitement. Le controle en fin de fonction garantit qu'aucune intention
# n'est oubliee le jour ou la source en ajoute une.
GROUPES = {
    "carte": [
        "card_arrival", "card_linking", "card_delivery_estimate", "card_not_working",
        "lost_or_stolen_card", "getting_virtual_card", "get_physical_card",
        "visa_or_mastercard", "disposable_card_limits", "compromised_card",
        "card_swallowed", "getting_spare_card", "order_physical_card",
        "virtual_card_not_working", "get_disposable_virtual_card", "activate_my_card",
        "card_about_to_expire", "card_acceptance", "contactless_not_working",
        "apple_pay_or_google_pay", "supported_cards_and_currencies",
    ],
    "paiement": [
        "pending_card_payment", "declined_card_payment", "card_payment_not_recognised",
        "reverted_card_payment?", "direct_debit_payment_not_recognised",
        "card_payment_wrong_exchange_rate", "exchange_rate", "fiat_currency_support",
        "exchange_via_app", "country_support",
    ],
    "virement": [
        "cancel_transfer", "transfer_not_received_by_recipient", "declined_transfer",
        "pending_transfer", "transfer_timing", "beneficiary_not_allowed",
        "receiving_money", "failed_transfer", "transfer_into_account",
        "balance_not_updated_after_bank_transfer",
    ],
    "retrait": [
        "pending_cash_withdrawal", "wrong_amount_of_cash_received",
        "declined_cash_withdrawal", "atm_support", "cash_withdrawal_not_recognised",
        "wrong_exchange_rate_for_cash_withdrawal",
    ],
    "rechargement": [
        "automatic_top_up", "pending_top_up", "top_up_limits", "top_up_reverted",
        "topping_up_by_card", "top_up_by_cash_or_cheque", "top_up_failed",
        "verify_top_up", "balance_not_updated_after_cheque_or_cash_deposit",
    ],
    "frais": [
        "extra_charge_on_statement", "card_payment_fee_charged",
        "top_up_by_bank_transfer_charge", "transfer_fee_charged", "exchange_charge",
        "top_up_by_card_charge", "cash_withdrawal_charge", "transaction_charged_twice",
    ],
    "remboursement": ["request_refund", "Refund_not_showing_up"],
    "compte": [
        "edit_personal_details", "why_verify_identity", "unable_to_verify_identity",
        "passcode_forgotten", "pin_blocked", "change_pin", "terminate_account",
        "age_limit", "verify_source_of_funds", "verify_my_identity",
        "lost_or_stolen_phone",
    ],
}

GRAINE = 42

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


def construire_rfm() -> pd.DataFrame:
    """Recence, frequence, montant : un client du detaillant par ligne.

    Les trois variables classiques de la segmentation client, et la table de
    la seance 3.2. `pays` est gardee pour pouvoir relire les segments obtenus
    a la lumiere des marches, sans entrer dans le calcul des distances.
    """
    ventes = pd.read_csv(RETAIL / "ventes.csv", parse_dates=["date"])
    clients = pd.read_csv(RETAIL / "clients.csv")
    ventes["ca"] = ventes["qte"] * ventes["prix"]
    fin = ventes["date"].max()

    rfm = ventes.groupby("client_id", as_index=False).agg(
        recence=("date", lambda s: (fin - s.max()).days),
        freq=("cmd_id", "nunique"),
        montant=("ca", "sum"),
    )
    rfm["montant"] = rfm["montant"].round(2)
    return rfm.merge(clients[["client_id", "pays"]], on="client_id")


def construire_produits() -> pd.DataFrame:
    """Un profil d'achat par reference, pour la partie 2 de la seance 3.2.

    On ne garde que les references vendues au moins dix fois : en dessous, le
    profil n'est pas un profil, c'est une anecdote.
    """
    ventes = pd.read_csv(RETAIL / "ventes.csv", parse_dates=["date"])
    clients = pd.read_csv(RETAIL / "clients.csv")
    ventes["ca"] = ventes["qte"] * ventes["prix"]
    vc = ventes.merge(clients[["client_id", "pays"]], on="client_id")

    profil = vc.groupby("prod_id", as_index=False).agg(
        nb_cmd=("cmd_id", "nunique"),
        qte=("qte", "sum"),
        ca=("ca", "sum"),
        prix=("prix", "mean"),
        pays=("pays", "nunique"),
        clients=("client_id", "nunique"),
    )
    profil = profil.query("nb_cmd >= 10").copy()

    # Part du chiffre d'affaires realisee au dernier trimestre. C'est la seule
    # variable decorrelee des autres : c'est elle qui fera un axe de
    # segmentation interessant plutot qu'un simple classement par taille.
    q4 = vc[vc["date"].dt.month.isin([10, 11, 12])].groupby("prod_id")["ca"].sum()
    profil["part_q4"] = (q4.reindex(profil["prod_id"]).fillna(0).to_numpy()
                         / profil["ca"]).round(3)

    profil["ca"] = profil["ca"].round(2)
    profil["prix"] = profil["prix"].round(2)
    return profil


def _charger_banking(fichier: str) -> pd.DataFrame:
    """Un des deux fichiers de banking77, avec ses intentions regroupees."""
    cache = ICI / f"_source_banking_{fichier}"
    if not cache.exists():
        print(f"Telechargement de banking77 ({fichier})...")
        urllib.request.urlretrieve(URL_BANKING + fichier, cache)
    brut = pd.read_csv(cache)

    vers_groupe = {intention: groupe
                   for groupe, intentions in GROUPES.items()
                   for intention in intentions}

    # Le controle qui compte : si la source ajoute une intention un jour,
    # elle deviendrait silencieusement NaN et disparaitrait des comptages.
    inconnues = set(brut["category"]) - set(vers_groupe)
    assert not inconnues, f"intentions non classees : {sorted(inconnues)}"

    brut["equipe"] = brut["category"].map(vers_groupe)
    return brut.rename(columns={"text": "message", "category": "intention"})


def construire_tickets() -> tuple:
    """Les deux lots de messages clients de l'atelier « The Inbox Problem ».

    `tickets.csv` sert les seances 3.3 et 3.4 de bout en bout : recherche
    semantique, carte des themes, classifieur, extraction par LLM.
    `tickets_nouveaux.csv` est le lot INEDIT du defi final — il vient du
    fichier de test de la source, donc d'aucun message deja vu.

    5 000 lignes et non 10 000 : c'est le volume qui tient en ~90 secondes
    d'embeddings sur un CPU Colab, et la seance entiere en depend.
    """
    train = _charger_banking("train.csv")
    test = _charger_banking("test.csv")

    tickets = (train.sample(5000, random_state=GRAINE)
               .sort_index()
               .reset_index(drop=True)[["message", "intention", "equipe"]])

    # Le lot du defi ne contient PAS la reponse : les equipes doivent la
    # produire. L'etiquette reste dans un fichier a part, cote intervenant.
    nouveaux = (test.sample(500, random_state=GRAINE)
                .sort_index()
                .reset_index(drop=True)[["message", "intention", "equipe"]])
    return tickets, nouveaux


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
    # kind="stable" : a horodatage egal, l'ordre des cmd_id issu du groupby
    # est conserve. Sans lui, l'ordre des lignes ex aequo depend de la version
    # de pandas — et l'ordre des lignes decide du decoupage
    # train_test_split de la seance 3.1, donc de toutes ses valeurs attendues.
    commandes = commandes[
        ["cmd_id", "date", "jour", "ca", "nart", "qte", "pays", "client_id"]
    ].sort_values("date", kind="stable").reset_index(drop=True)

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
    tickets, nouveaux = construire_tickets()

    for nom, table in [("commandes.csv", commandes),
                       ("churn.csv", churn),
                       ("clients_rfm.csv", construire_rfm()),
                       ("produits_profil.csv", construire_produits()),
                       ("tickets.csv", tickets),
                       ("tickets_nouveaux.csv", nouveaux)]:
        chemin = ICI / nom
        table.to_csv(chemin, index=False)
        taille = chemin.stat().st_size / 1e6
        print(f"{nom:<16} {len(table):>6} lignes  {taille:5.2f} Mo")

    print(f"jours presents : {commandes['jour'].nunique()} (samedi absent, comme attendu)")
    print(f"taux de resiliation : {100 * churn['churn'].mean():.1f} %")
    print(f"equipes de routage : {tickets['equipe'].nunique()}")
    print(tickets["equipe"].value_counts().to_string())


if __name__ == "__main__":
    main()
