def get_user_input(encoder):
    print("---------input---------")
    print("Veuillez entrer la superficie (en m²) de votre bien immobilier :")
    area = int(input())
    print("Veuillez entrer le nom de la ville : ")
    city_name = input().upper() 
    print("Veuillez entrer le code postal : ")
    postal_code = input() 

    # Encodage des valeurs city_name et postal_code
    encoded_values = encoder.transform([[city_name, postal_code]])

    city_name_encoded = encoded_values[0][0]
    postal_code_encoded = encoded_values[0][1]

    return area, city_name_encoded, postal_code_encoded