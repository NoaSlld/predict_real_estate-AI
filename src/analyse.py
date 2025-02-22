import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_csv("./data/full_cleaned.csv", dtype={'nom_commune': 'object', 'code_departement': 'string'}, low_memory=False)


def repartitionMontantVentes():
    df_filtered = df[df['valeur_fonciere'] <= 1e6]

    sns.histplot(df_filtered['valeur_fonciere'], bins=50, kde=False)
    plt.xlim(0, 1e6)  # Limites de l'axe des X entre 0 et 1 million €

    plt.xlabel("Prix de la transaction (€)")
    plt.ylabel("Nombre de biens")
    plt.title("Répartition des valeurs foncières (0 à 1 000 000 €)")

    plt.show()


def evolutionPricesOverYears():
    #  évolution du prix moyen et median par année
    prix_par_annee = df.groupby('annee')['valeur_fonciere'].agg(['mean', 'median'])

    plt.figure(figsize=(10,5))
    sns.lineplot(data=prix_par_annee, x=prix_par_annee.index, y='mean', label='Prix moyen', marker='o')
    sns.lineplot(data=prix_par_annee, x=prix_par_annee.index, y='median', label='Prix médian', marker='s')
    plt.xlabel('Année')
    plt.ylabel('Prix (€)')
    plt.title("Évolution du prix moyen et médian de l'immobilier")
    plt.legend()
    plt.grid(True)
    plt.show()


def mostExpensiveSquareMetre():
    # communes dont le prix au m² est le plus élevé
    df['prix_m2'] = df['valeur_fonciere'] / df['surface_reelle_bati']
    prix_par_commune = df.groupby('nom_commune')['prix_m2'].median().sort_values(ascending=False).head(10)

    plt.figure(figsize=(12,6))
    sns.barplot(x=prix_par_commune.values, y=prix_par_commune.index, palette='Reds_r')
    plt.xlabel('Prix médian au m² (€)')
    plt.ylabel('Commune')
    plt.title("Top 10 des communes les plus chères en prix au m²")
    plt.show()


def biggestPriceIncreaseOverYears():

    # Prix du m² moyen par commune et par année
    df['prix_m2'] = df['valeur_fonciere'] / df['surface_reelle_bati']
    prix_par_commune_annee = df.groupby(['annee', 'nom_commune'])['prix_m2'].mean().unstack()

    premiere_annee = prix_par_commune_annee.index.min()
    derniere_annee = prix_par_commune_annee.index.max()

    # Filtrer les communes ayant des données pour les deux années
    prix_par_commune_annee = prix_par_commune_annee.dropna(axis=1, how='any')

    evolution = prix_par_commune_annee.loc[derniere_annee] - prix_par_commune_annee.loc[premiere_annee]

    # 10 communes avec la plus forte augmentation de prix moyen
    top_communes_hausse = evolution.sort_values(ascending=False).head(10)

    plt.figure(figsize=(12,6))
    sns.barplot(x=top_communes_hausse.index, y=top_communes_hausse.values, palette='Greens_r')
    plt.xticks(rotation=45)
    plt.xlabel('Commune')
    plt.ylabel('Augmentation du prix moyen au m² (€)')
    plt.title("Top 10 des communes avec la plus forte hausse du prix moyen au m²")
    plt.show()

