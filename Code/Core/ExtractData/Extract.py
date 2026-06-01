"""
███████╗██╗░░██╗████████╗██████╗░░█████╗░░█████╗░████████╗
██╔════╝╚██╗██╔╝╚══██╔══╝██╔══██╗██╔══██╗██╔══██╗╚══██╔══╝
█████╗░░░╚███╔╝░░░░██║░░░██████╔╝███████║██║░░╚═╝░░░██║░░░
██╔══╝░░░██╔██╗░░░░██║░░░██╔══██╗██╔══██║██║░░██╗░░░██║░░░
███████╗██╔╝╚██╗░░░██║░░░██║░░██║██║░░██║╚█████╔╝░░░██║░░░
╚══════╝╚═╝░░╚═╝░░░╚═╝░░░╚═╝░░╚═╝╚═╝░░╚═╝░╚════╝░░░░╚═╝░░░



"""

############// Selecionando as imagens!! //#####################################


# Diretório com as imagens
#os.chdir(image_dir)
imgs_path = '../../../DataSets/Flowers/imgs'  # Caminho onde as imagens estão armazenadas
images = natsort.natsorted(os.listdir(imgs_path))  # Lista ordenada de imagens
DataSet = "Flowers"

# Inicialização da lista de features e arquivo de saída
#features = []  # Lista para armazenar as features extraídas
dataset_elements = []  # Lista auxiliar para os nomes das imagens



############// Preprocessando!! //#####################################


# Paths das imagens
image_paths = []

ini_p = time.time()
for i, img in enumerate(images):
    if ".jpg" not in img:  # Ignora arquivos que não sejam imagens .jpg
        continue

    # Salva o nome da imagem no arquivo de texto
    dataset_elements.append(img)

    # Define o caminho completo da imagem e adiciona à lista
    img_path = os.path.join(imgs_path, img)
    image_paths.append(img_path)

    # Log a cada 250 imagens processadas
    if i % 250 == 0:
        print(f"{i} images collected!")

end_p = time.time()



############// Inferencia do modelo //#####################################


features = dinov2_inference(model_name, image_paths)



# Salvando as features
emb_path = "../../../DataSets/Emb/" + dataset_name + "_emb.npy"

np.save(emb_path, features)
print("Done!")





############// Listas Ranqueadas!! //#####################################


import numpy as np
from sklearn.neighbors import BallTree


def run_ball_tree(features, k=100):
    """
    Constrói uma estrutura BallTree a partir das features e retorna os rankings dos vizinhos mais próximos.
    """


    # Verifica se as features são válidas
    if not isinstance(features, np.ndarray):
        raise ValueError("As 'features' devem ser um array do tipo numpy.ndarray.")
    if features.ndim != 2:
        raise ValueError("As 'features' devem ser um array 2D no formato (n_samples, n_features).")

    # Cria a estrutura BallTree
    tree = BallTree(features)

    # Realiza a consulta para encontrar os k vizinhos mais próximos
    _, rks = tree.query(features, k=k)

    return rks


rks = run_ball_tree(features)



############// Exportando as listas Ranqueadas!! //#####################################


import json

# Convert NumPy array to JSON and export
os.chdir("../Runs")
with open(model_name + "_output.json", "w") as json_file:
    # Format the JSON with proper indentation for readability
    json.dump(rks.tolist(), json_file, indent=4)
    # Ensure the file is properly closed and flushed
    json_file.flush()
    print(f"Data successfully exported to Runs/{model_name}_output.json")


############// Plotting the graph!! //#####################################
os.chdir("..")

rks_path = "./Runs/" + model_name + "_output.json"
with open(rks_path, "r") as f:
    print("FOUND FILE!!")
    rankings = json.load(f)

# Convert to numpy array
rankings = np.array(rankings)

os.chdir("./Plots")

import os
from Graph import plot_and_export

k_list = [10, 20, 30, 40, 60, 80]

plot_and_export(rankings, k_list, model_name)


#------------// End of the program //--------------------------
