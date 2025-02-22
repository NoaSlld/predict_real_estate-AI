import pandas as pd

# On lit tous les fichiers csv dans l'ordre des années
df_list = [pd.read_csv(f'../data/full_{year}.csv', dtype={'code_departement': str}) for year in range(2017, 2024)]
df = pd.concat(df_list, ignore_index=True)  # ignore_index évite les doublons d'index inutiles
del df_list  # Supprime la liste pour libérer la mémoire

# on enlève les colonnes avec plus de 60% de valeurs manquantes
df = df.drop(columns=['lot5_surface_carrez', 'lot4_surface_carrez', 'ancien_id_parcelle', 'lot5_numero', 'numero_volume', 'lot3_surface_carrez', 
                       'lot4_numero',  'ancien_nom_commune', 'ancien_code_commune', 'lot3_numero', 'lot2_surface_carrez', 'code_nature_culture_speciale', 
                       'nature_culture_speciale', 'adresse_suffixe', 'lot2_numero', 'lot1_surface_carrez', 'lot1_numero',  
                       'nombre_pieces_principales', 'type_local', 'code_type_local', 'adresse_numero', 'surface_terrain', 'nature_culture', 
                       'adresse_code_voie', 'id_parcelle', 'id_mutation', 'numero_disposition', 'code_commune', 'nombre_lots', 'latitude', 'longitude', 'code_nature_culture'])


# Suppression des doublons
print(f"Nombre de lignes avant suppression des doublons : {df.shape[0]}")
df = df.drop_duplicates()
print(f"Nombre de lignes après suppression des doublons : {df.shape[0]}")

# ------Nettoyage des valeurs aberrantes------
# Suppression des valeurs foncières trop basses ou trop élevées (exemple : < 1000€ ou > 10M€)
df = df[(df['valeur_fonciere'] >= 1000) & (df['valeur_fonciere'] <= 1e7)]
# Suppression des surfaces anormalement élevées (> 1000m² pour du résidentiel)
df = df[(df['surface_reelle_bati'] > 3) & (df['surface_reelle_bati'] < 1000)]

# ------Formatage------
# Nettoyage des espaces superflus et mise en majuscule des noms de commune
df['nom_commune'] = df['nom_commune'].str.strip().str.upper()
# Conversion de la date en datetime
df['date_mutation'] = pd.to_datetime(df['date_mutation'], errors='coerce')


# ------ Remplacement des NaN par la moyenne des biens similaires ------
def impute_surface(row, df):
    """ Remplace NaN par la moyenne des biens de la même ville avec une superficie similaire (±10%) """
    if pd.notna(row['surface_reelle_bati']):
        return row['surface_reelle_bati']  # garde la valeur existante
    
    commune = row['nom_commune']
    surface_min = row['surface_reelle_bati'] * 0.9
    surface_max = row['surface_reelle_bati'] * 1.1
    
    voisins = df[(df['nom_commune'] == commune) & 
                 (df['surface_reelle_bati'].between(surface_min, surface_max))]
    
    if not voisins.empty:
        return int(round(voisins['surface_reelle_bati'].mean()))
    else:
        return None  # on laisse NaN si aucun voisin trouvé

# Application du remplacement
df['surface_reelle_bati'] = df.apply(lambda row: impute_surface(row, df), axis=1)

# Suppression des lignes avec valeurs manquantes
df.dropna(inplace=True)

# Enregistrement du fichier nettoyé
df.to_csv('../data/full_cleaned.csv', index=False)
df = pd.read_csv('../data/full_cleaned.csv', dtype={'code_departement': str})

print(f"Nombre de lignes après le nettoyage terminé : {df.shape[0]}")
print("Valeurs manquantes après traitement :")
print(df.isna().sum())
