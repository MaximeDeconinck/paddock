"""Fictitious September 2026 bank statements for the paddock demo.

Writes demo/workspace/data/{compte-courant,compte-joint,carte}.csv (date;libelle;montant).
Every name and amount is invented. Fixed seed: the files are reproducible.
Planted findings: the same streaming subscription paid from two accounts,
a card payment charged twice, and one unusually large expense.
"""

import random
from pathlib import Path

random.seed(2026)
OUT = Path(__file__).parent / "workspace" / "data"


def day(d):
    return f"2026-09-{d:02d}"


def amount(lo, hi):
    return round(random.uniform(lo, hi), 2)


def write(name, rows):
    rows.sort(key=lambda r: r[0])
    text = "date;libelle;montant\n" + "".join(
        f"{d};{label};{str(f'{v:.2f}').replace('.', ',')}\n" for d, label, v in rows
    )
    (OUT / name).write_text(text, encoding="utf-8")


def courant():
    rows = [
        (day(1), "VIR SALAIRE NOVALYS SAS", 2840.00),
        (day(2), "PRLV LOYER SCI DES TILLEULS", -890.00),
        (day(3), "PRLV EDF ENERGIE BLEUE", -74.30),
        (day(4), "PRLV FORFAIT MOBILE ONDA", -19.99),
        (day(4), "PRLV BOX FIBRE ONDA", -32.99),
        (day(5), "PRLV ABONNEMENT STREAMFLIX", -13.49),
        (day(6), "PRLV ASSURANCE HABITATION MUTUELLE VERTE", -21.40),
        (day(8), "PRLV SALLE DE SPORT FITPARK", -39.90),
        (day(10), "VIR EPARGNE LIVRET A", -300.00),
        (day(12), "PRLV MUTUELLE SANTE HARMONIA", -48.60),
        (day(15), "PRLV ASSURANCE AUTO PRUDENCIA", -56.20),
        (day(18), "GARAGE DU CENTRE REPARATION EMBRAYAGE", -1240.00),
        (day(20), "VIR REMBOURSEMENT LUCAS", 45.00),
        (day(24), "PRLV MUSIQUE SOUNDWAVE", -10.99),
        (day(27), "PRLV CREDIT CONSO FINAGO", -142.00),
        (day(28), "FRAIS TENUE DE COMPTE", -2.00),
    ]
    for d in random.sample(range(1, 31), 8):
        rows.append((day(d), "RETRAIT DAB", -float(random.choice([20, 40, 50, 60]))))
    return rows


def joint():
    rows = [
        (day(1), "VIR ALIMENTATION COMPTE JOINT", 900.00),
        (day(1), "VIR ALIMENTATION COMPTE JOINT CONJOINT", 900.00),
        (day(5), "PRLV ABONNEMENT STREAMFLIX", -13.49),
        (day(7), "PRLV CANTINE ECOLE JULES FERRY", -86.40),
        (day(9), "PRLV CRECHE LES PETITS PAS", -310.00),
        (day(14), "PRLV EAU SAUR", -38.75),
        (day(21), "PRLV ASSURANCE SCOLAIRE", -12.00),
    ]
    shops = [
        ("CARREFOUR MARKET", 35, 140),
        ("LECLERC DRIVE", 60, 180),
        ("BIOCOOP", 18, 55),
        ("BOULANGERIE MAISON PAIN", 4, 14),
        ("PHARMACIE DU MARCHE", 7, 38),
        ("PICARD", 15, 45),
        ("PRIMEUR LES 4 SAISONS", 9, 30),
    ]
    for _ in range(38):
        name, lo, hi = random.choice(shops)
        rows.append((day(random.randint(1, 30)), f"CB {name}", -amount(lo, hi)))
    rows.append((day(16), "CB DECATHLON RENTREE", -127.80))
    rows.append((day(3), "CB FOURNITURES SCOLAIRES PAPETERIE", -64.15))
    return rows


def carte():
    rows = []
    places = [
        ("UBER EATS", 14, 38, 7),
        ("DELIVEROO", 16, 34, 5),
        ("SNCF CONNECT", 25, 89, 3),
        ("TOTAL ENERGIES STATION", 45, 78, 4),
        ("AMAZON MARKETPLACE", 9, 75, 6),
        ("CAFE DE LA GARE", 3, 9, 8),
        ("RESTAURANT LE PETIT ZINC", 22, 64, 4),
        ("CINEMA PATHE", 11, 24, 2),
        ("ZARA", 25, 90, 2),
        ("FNAC", 15, 60, 2),
        ("PARKING INDIGO", 3, 16, 5),
        ("LIME TROTTINETTE", 2, 7, 6),
    ]
    for name, lo, hi, n in places:
        for _ in range(n):
            rows.append((day(random.randint(1, 30)), f"CB {name}", -amount(lo, hi)))
    # Planted duplicate: the same payment charged twice on the same day.
    rows.append((day(19), "CB HOTEL IBIS LYON PART DIEU", -94.00))
    rows.append((day(19), "CB HOTEL IBIS LYON PART DIEU", -94.00))
    rows.append((day(11), "CB APPLE.COM/BILL ICLOUD", -2.99))
    return rows


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    write("compte-courant.csv", courant())
    write("compte-joint.csv", joint())
    write("carte.csv", carte())
    for f in sorted(OUT.glob("*.csv")):
        print(f.name, sum(1 for _ in f.open()) - 1, "rows")
