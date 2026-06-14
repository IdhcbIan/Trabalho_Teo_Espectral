"""
-------------------------------------------

Trabalho De Teoria Espectral.

Neste codigo usamos os embeddings gerados pelo Extract.py.
Depois disso, o BallTree cria uma lista ranqueada de vizinhos mais
similares para cada imagem.

A ideia deste script eh:

    imagens -> ranking KNN -> grafo ponderado W
    -> Laplaciano normalizado L
    -> autovetores de L
    -> KMeans no espaco laplaciano
    -> avaliacao usando os rotulos reais das flores.

Importante:

    O algoritmo nao usa os rotulos para criar os clusters.
    Os rotulos aparecem apenas no final, para medir se os clusters
    encontrados batem com as classes reais.

-------------------------------------------
"""


import csv
import json
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from Aux import (
    plot_first_two_embedding_coordinates,
    plot_graph_by_cluster,
    plot_graph_by_correctness,
)


ROOT_DIR = Path(__file__).resolve().parents[3]
DATASETS_DIR = ROOT_DIR / "DataSets"

# -------------------------------------------------------------------------
# Configuracao fixa do experimento.
# -------------------------------------------------------------------------

# Active experiment. Comment/uncomment this small block when running by hand.
#DATASET_NAME = "Flowers"; OUTPUT_NAME = "Flowers_dinov2_vits14"; LABEL_MODE = "fixed"; NUM_CLUSTERS = 17; SAMPLES_PER_CLASS = 80; OUTPUT_DIR = DATASETS_DIR / "Laplacian"
DATASET_NAME = "CUB_Cleaned50"; OUTPUT_NAME = "CUB_Cleaned50_dinov2_vits14"; LABEL_MODE = "manifest"; NUM_CLUSTERS = 50; SAMPLES_PER_CLASS = None; OUTPUT_DIR = DATASETS_DIR / "Laplacian"
#DATASET_NAME = "Flowers"; OUTPUT_NAME = "AlexNet_Flowers"; LABEL_MODE = "fixed"; NUM_CLUSTERS = 17; SAMPLES_PER_CLASS = 80; OUTPUT_DIR = DATASETS_DIR / "Laplacian" / "AlexNet"

RANKING_PATH = DATASETS_DIR / "Runs" / f"{OUTPUT_NAME}_output.json"
LABELS_PATH = DATASETS_DIR / DATASET_NAME / "manifest.csv"

K = 20
SEED = 42

LAYOUT_K = 0.34
LAYOUT_ITERATIONS = 180
LAYOUT_SCALE = 1.35


from Aux import load_labels_from_manifest


def load_labels(n_samples):
    if LABEL_MODE == "manifest":
        return load_labels_from_manifest(LABELS_PATH, n_samples)

    if LABEL_MODE == "fixed":
        expected = NUM_CLUSTERS * SAMPLES_PER_CLASS
        if n_samples != expected:
            raise ValueError(f"Expected {expected} samples, found {n_samples}.")
        return np.repeat(np.arange(NUM_CLUSTERS), SAMPLES_PER_CLASS)

    raise ValueError(f"Unknown LABEL_MODE: {LABEL_MODE}")

def main():
    """
    Executa o experimento configurado nas variaveis fixas no topo do arquivo.
    """

    # Garante que a pasta de saida existe antes de salvar JSON, NPY e PNG.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------
    # 1. Carregando o ranking KNN do JSON.
    # ---------------------------------------------------------------------
    print("\n[1] Loading nearest-neighbor rankings")

    """
    O arquivo de ranking eh uma lista em JSON.


    rankings[i] = [self, vizinho_1, vizinho_2, vizinho_3, ...]
    """

    with open(RANKING_PATH, "r") as f:
        rankings = np.array(json.load(f), dtype=int)

    if rankings.ndim != 2:
        raise ValueError("Ranking JSON must be a 2D list.")

    n_samples, n_ranked_neighbors = rankings.shape
    print(f"Rankings shape: {rankings.shape}")

    """
    Verificando se o K que escolhemos eh viavel dentro do k maximo no JSON.
    """

    if K >= n_ranked_neighbors:
        raise ValueError(
            f"k={K} is too large. Ranking file has only {n_ranked_neighbors} entries per row."
        )

    # ---------------------------------------------------------------------
    # 2. Construindo nosso grafo ponderado.
    # ---------------------------------------------------------------------
    print("\n[2] Building weighted KNN graph W")

    """
    Vamos construir W a partir das nossas listas de ranking KNN.

    W eh a matriz de adjacencia ponderada:

        W[i, j] = peso da aresta da imagem i para a imagem j

    Na nossa lista temos, para k=5 por exemplo:

        [Self, 1_Sim, 2_Sim, 3_Sim, 4_Sim, 5_Sim]

    Usamos a posicao no ranking para definir o peso:

        rank 1 neighbor -> weight 1.0
        rank 2 neighbor -> weight 0.5
        rank 3 neighbor -> weight 0.333...
        rank r neighbor -> weight 1/r

    Basicamente:

        peso = 1 / numero_no_ranking

    Assim, vizinhos mais parecidos influenciam mais o grafo.
    """

    W = np.zeros((n_samples, n_samples), dtype=float)

    for i in range(n_samples):
        # Pulamos a primeira posicao porque normalmente eh a propria imagem.
        neighbors = rankings[i, 1 : K + 1]

        for rank, j in enumerate(neighbors, start=1):
            if i == j:
                continue

            # Se o mesmo vizinho aparecer de alguma forma repetida, ficamos
            # com o maior peso encontrado para aquela direcao i -> j.
            W[i, j] = max(W[i, j], 1.0 / rank)

    """
    Agora vamos transformar W em um grafo nao direcionado.

    Antes:

        W[i, j] significa que i escolheu j como vizinho.

    Mas para o Laplaciano do grafo queremos uma relacao nao direcionada:

        i conectado com j

    Por isso fazemos:

        W = W + W.T

    Isso faz com que W fique simetrica e as similaridades sejam somadas.

    Exemplo:

        a -> b: 1 / 1 = 1.0
        b -> a: 1 / 2 = 0.5

    Entao a aresta final entre a e b tem peso:

        1.0 + 0.5 = 1.5

    Ou seja: vizinhos reciprocos ficam mais fortes no grafo.
    """

    W = W + W.T

    """
    Criamos tambem um objeto grafo da biblioteca NetworkX.

    Vamos usar a matriz W para o Laplaciano e os autovetores.
    Usamos o grafo NetworkX eh usado principalmente para visualizacao e exportacao.
    """

    graph = nx.Graph()
    graph.add_nodes_from(range(n_samples))

    # Pegamos apenas a parte triangular superior para nao adicionar a mesma
    # aresta duas vezes, ja que W agora eh simetrica.
    rows, cols = np.nonzero(np.triu(W, k=1))

    for i, j in zip(rows, cols):
        graph.add_edge(int(i), int(j), weight=float(W[i, j]))

    print(f"Nodes: {graph.number_of_nodes()}")
    print(f"Edges: {graph.number_of_edges()}")

    # ---------------------------------------------------------------------
    # 3. Construindo o Laplaciano normalizado explicitamente.
    # ---------------------------------------------------------------------
    print("\n[3] Computing normalized graph Laplacian")

    """
    Queremos calcular o Laplaciano normalizado:

        L = I - D^(-1/2) W D^(-1/2)

    Onde:

        W -> matriz de adjacencia ponderada.
        D -> matriz diagonal dos graus.
        I -> matriz identidade.

    O grau de um no i eh a soma dos pesos das arestas ligadas a ele:

        d_i = soma_j W[i, j]

    Como D eh diagonal, nao precisamos montar a matriz D inteira.
    Basta calcular um vetor com:

        d_inv_sqrt[i] = 1 / sqrt(d_i)

    Depois usamos broadcasting do NumPy para fazer:

        D^(-1/2) W D^(-1/2)

    Na pratica:

        d_inv_sqrt[:, None] multiplica as linhas.
        d_inv_sqrt[None, :] multiplica as colunas.

    Essa normalizacao evita que vertices com grau muito alto dominem
    completamente a estrutura laplaciana do grafo.
    """

    degrees = W.sum(axis=1)

    d_inv_sqrt = np.zeros_like(degrees)
    nonzero_degrees = degrees > 0
    d_inv_sqrt[nonzero_degrees] = 1.0 / np.sqrt(degrees[nonzero_degrees])

    normalized_adjacency = d_inv_sqrt[:, None] * W * d_inv_sqrt[None, :]
    L = np.eye(n_samples) - normalized_adjacency

    # ---------------------------------------------------------------------
    # 4. Pegando os menores autovetores de L.
    # ---------------------------------------------------------------------
    print("\n[4] Computing smallest eigenvectors")

    """
    Na propagacao pelo Laplaciano, a informacao dos clusters aparece nos
    autovetores associados aos menores autovalores do Laplaciano.

    Intuicao:

        se o grafo tem comunidades bem separadas,
        os menores autovetores conseguem representar essas comunidades
        como coordenadas mais faceis de separar.

    Usamos eigsh com:

        which="SM"

    para pedir os menores autovalores em modulo.
    """

    eigenvalues, U = eigsh(L, k=NUM_CLUSTERS, which="SM")

    """
    Ordenamos os autovalores e autovetores.

    Isso deixa o resultado estavel e mais facil de interpretar:

        eigenvalues[0] -> menor autovalor
        U[:, 0]        -> autovetor correspondente
    """
    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    U = U[:, order]

    print(f"U shape: {U.shape}")

    # ---------------------------------------------------------------------
    # 5. Normalizando cada linha de U.
    # ---------------------------------------------------------------------
    print("\n[5] Row-normalizing eigenvectors")

    """
    Cada linha de U representa uma imagem no espaco laplaciano.

    Por exemplo:

        U[i] = coordenadas espectrais da imagem i

    Normalizamos cada linha para tamanho 1:

        Y[i] = U[i] / ||U[i]||

    Esse passo eh comum no algoritmo de Ng-Jordan-Weiss.
    Ele faz o KMeans olhar mais para a direcao dos pontos no espaco
    laplaciano, e nao apenas para o tamanho dos vetores.
    """

    row_norms = np.linalg.norm(U, axis=1, keepdims=True)
    row_norms[row_norms == 0] = 1.0
    Y = U / row_norms

    # ---------------------------------------------------------------------
    # 6. Rodando KMeans no espaco laplaciano.
    # ---------------------------------------------------------------------
    print("\n[6] Running KMeans")

    """
    Os autovetores nao dao rotulos discretos diretamente.

    Eles colocam cada imagem em um novo espaco, chamado espaco laplaciano.
    Nesse espaco, imagens que pertencem a mesma comunidade do grafo tendem
    a ficar proximas.

    O KMeans entra aqui para transformar essas coordenadas continuas em
    grupos discretos:

        imagem i -> cluster c

    Portanto:

        KMeans nao esta clusterizando os embeddings originais do DINOv2.
        KMeans esta clusterizando as linhas de Y, que vieram do Laplaciano.
    """

    clusters = KMeans(
        n_clusters=NUM_CLUSTERS,
        n_init=50,
        random_state=SEED,
    ).fit_predict(Y)

    # ---------------------------------------------------------------------
    # 7. Avaliando os clusters com os rotulos reais.
    # ---------------------------------------------------------------------
    print("\n[7] Evaluating clusters")

    """
    Aqui usamos os rotulos reais apenas para avaliacao.

    Para o CUB_Cleaned50, as classes nao tem sempre o mesmo numero de
    imagens. Por isso carregamos os rotulos do manifest criado junto com o
    dataset limpo.
    """

    labels = load_labels(n_samples)

    """
    O KMeans escolhe nomes arbitrarios para os clusters.

    Exemplo:

        cluster 0 pode corresponder a classe 7
        cluster 1 pode corresponder a classe 3

    Entao nao podemos comparar diretamente:

        clusters == labels

    Primeiro montamos uma matriz de confusao:

        confusion[cluster_id, class_id]

    Ela conta quantas imagens do cluster_id pertencem a class_id.
    """

    confusion = np.zeros((NUM_CLUSTERS, NUM_CLUSTERS), dtype=int)

    for cluster_id in range(NUM_CLUSTERS):
        for class_id in range(NUM_CLUSTERS):
            confusion[cluster_id, class_id] = np.sum(
                (clusters == cluster_id) & (labels == class_id)
            )

    """
    Depois usamos o algoritmo Hungaro para achar o melhor pareamento:

        cluster -> classe real

    O linear_sum_assignment minimiza custo.
    Como queremos maximizar acertos, transformamos a matriz de contagens em
    uma matriz de custo:

        custo = confusion.max() - confusion

    Assim, quanto maior a sobreposicao entre cluster e classe, menor o custo.
    """

    rows, cols = linear_sum_assignment(confusion.max() - confusion)
    cluster_to_class = {int(row): int(col) for row, col in zip(rows, cols)}
    mapped_labels = np.array([cluster_to_class[c] for c in clusters])

    """
    Agora sim conseguimos calcular metricas.

    accuracy:
        porcentagem de imagens cujo cluster mapeado bate com o rotulo real.

    ARI:
        mede similaridade entre particoes, corrigindo efeito do acaso.

    NMI:
        mede informacao compartilhada entre clusters e classes reais.
    """

    accuracy = np.mean(mapped_labels == labels)
    ari = adjusted_rand_score(labels, clusters)
    nmi = normalized_mutual_info_score(labels, clusters)

    print(f"Accuracy after cluster-label matching: {accuracy:.4f}")
    print(f"Adjusted Rand Index: {ari:.4f}")
    print(f"Normalized Mutual Information: {nmi:.4f}")
    print(f"Eigenvalues: {[round(float(v), 6) for v in eigenvalues]}")
    print(f"Cluster mapping: {cluster_to_class}")

    # ---------------------------------------------------------------------
    # 8. Calculando layout do grafo e exportando JSON.
    # ---------------------------------------------------------------------
    print("\n[8] Computing graph layout and exporting JSON")

    """
    O spring_layout calcula posicoes 2D para desenhar o grafo.

    Ele nao muda o resultado do clustering.
    Ele serve apenas para visualizacao e exportacao:

        node_id -> posicao [x, y]

    Usamos weight="weight" para o layout considerar pesos das arestas.
    """

    positions = nx.spring_layout(
        graph,
        k=LAYOUT_K,
        iterations=LAYOUT_ITERATIONS,
        seed=SEED,
        weight="weight",
        scale=LAYOUT_SCALE,
    )

    """
    O JSON exportado guarda:

        nodes:
            posicao do no
            rotulo real
            cluster encontrado
            rotulo mapeado
            se acertou ou errou

        edges:
            lista de vizinhos no grafo

        graph_info:
            informacoes gerais do experimento
    """

    graph_json = {
        "nodes": {},
        "edges": {},
        "graph_info": {
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
            "k": K,
            "dataset": DATASET_NAME,
            "output_name": OUTPUT_NAME,
            "method": "laplaciano_propagacao",
            "eigenvalues": [float(v) for v in eigenvalues],
        },
    }

    for node_id in graph.nodes:
        graph_json["nodes"][str(node_id)] = {
            "position": [
                float(positions[node_id][0]),
                float(positions[node_id][1]),
            ],
            "attributes": {
                "true_label": int(labels[node_id]),
                "cluster": int(clusters[node_id]),
                "mapped_label": int(mapped_labels[node_id]),
                "correct": bool(labels[node_id] == mapped_labels[node_id]),
            },
        }

    for node_id in graph.nodes:
        graph_json["edges"][str(node_id)] = [
            int(neighbor)
            for neighbor in graph.neighbors(node_id)
        ]

    output_stem = f"{OUTPUT_NAME}_laplacian_label_propagation_k{K}_c{NUM_CLUSTERS}"

    with open(OUTPUT_DIR / f"{output_stem}.json", "w") as f:
        json.dump(graph_json, f, indent=2)

    # ---------------------------------------------------------------------
    # 9. Salvando arrays e plots.
    # ---------------------------------------------------------------------
    print("\n[9] Saving outputs")

    """
    Salvamos os resultados em arquivos separados para facilitar analise.

    clusters.npy:
        cluster bruto do KMeans para cada imagem.

    mapped_labels.npy:
        cluster convertido para a melhor classe real correspondente.

    embedding.npy:
        coordenadas espectrais Y usadas pelo KMeans.

    eigenvalues.npy:
        autovalores escolhidos do Laplaciano.

    confusion.npy:
        matriz cluster x classe antes do mapeamento Hungaro.
    """

    np.save(OUTPUT_DIR / f"{output_stem}_clusters.npy", clusters)
    np.save(OUTPUT_DIR / f"{output_stem}_mapped_labels.npy", mapped_labels)
    np.save(OUTPUT_DIR / f"{output_stem}_embedding.npy", Y)
    np.save(OUTPUT_DIR / f"{output_stem}_eigenvalues.npy", eigenvalues)
    np.save(OUTPUT_DIR / f"{output_stem}_confusion.npy", confusion)

    """
    Tambem salvamos tres figuras:

        1. Grafo colorido por cluster encontrado.
        2. Grafo colorido por acerto/erro depois do mapeamento.
        3. Scatter plot das duas primeiras coordenadas espectrais.
    """

    plot_graph_by_cluster(
        graph,
        positions,
        clusters,
        OUTPUT_DIR / f"{output_stem}.png",
    )
    plot_graph_by_correctness(
        graph,
        positions,
        labels,
        mapped_labels,
        OUTPUT_DIR / f"{output_stem}_correctness.png",
    )
    plot_first_two_embedding_coordinates(
        Y,
        clusters,
        OUTPUT_DIR / f"{output_stem}_embedding.png",
    )

    print(f"Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
