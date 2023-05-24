import networkx as nx
import matplotlib.pyplot as plt
from lyricsgenius import Genius
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import community
import matplotlib.cm as cm
import numpy as np
import pandas as pd


def load_correlation_graph(words):
    graph = nx.Graph()

    # Percorre todas as palavras
    for i in range(len(words)):
        word = words[i]

        # Adicione o nó ao grafo se ainda não existir
        if word not in graph:
            graph.add_node(word)

        # Verifique as palavras vizinhas dentro de um contexto definido
        context = 3  # Quantas palavras antes e depois considerar como vizinhas
        neighbors = words[max(0, i - context):i] + words[i+1:i+context+1]

        # Adicione as arestas entre a palavra atual e suas palavras vizinhas
        for neighbor in neighbors:
            if neighbor not in graph:
                graph.add_node(neighbor)
            graph.add_edge(word, neighbor)

    return graph


def load_louvain_cliques_graph_nodes(cliques):
    louvain_clique_graph = nx.Graph()
    louvain_clique_graph.add_nodes_from(range(len(cliques)))
    return louvain_clique_graph


def graph_text_to_graph_index(graph, cliques):
    # Mapear os nós para seus respectivos índices
    node_map = {node: i for i, node in enumerate(graph.nodes())}

    # Criar uma lista de cliques mapeados para os seus respectivos índices
    graph_indexes = [[node_map[node] for node in clique] for clique in cliques]

    return graph_indexes


def load_louvain_cliques_graph_edges(graph, louvain_clique_graph, cliques):
    graph_indexes = graph_text_to_graph_index(graph, cliques)
    # Adicionar as arestas ao grafo das cliques
    for i in range(len(cliques)):
        for j in range(i + 1, len(cliques)):
            if len(set(graph_indexes[i]) & set(graph_indexes[j])) > 0:
                louvain_clique_graph.add_edge(i, j)
    return louvain_clique_graph


def load_correlation_graph(words):
    graph = nx.Graph()

    # Percorre todas as palavras
    for i in range(len(words)):
        word = words[i]

        # Adicione o nó ao grafo se ainda não existir
        if word not in graph:
            graph.add_node(word)

        # Verifique as palavras vizinhas dentro de um contexto definido
        context = 3  # Quantas palavras antes e depois considerar como vizinhas
        neighbors = words[max(0, i - context):i] + words[i+1:i+context+1]

        # Adicione as arestas entre a palavra atual e suas palavras vizinhas
        for neighbor in neighbors:
            if neighbor not in graph:
                graph.add_node(neighbor)
            graph.add_edge(word, neighbor)

    return graph


def show_louvain_partition_report(partition):
    # Criar uma lista para armazenar os nós em cada setor
    setor_nodes = [[] for _ in range(max(partition.values()) + 1)]

    # Agrupar os nós em cada setor com base na partição
    for node, setor in partition.items():
        setor_nodes[setor].append(node)

    # Exibir os nós em cada setor
    for setor, nodes in enumerate(setor_nodes):
        print(f"Setor {setor+1}: {nodes}")


def load_all_lyrics():
    # Acesso a API do Genius
    token = 'HpJ0pnH7fP-sc9GodGLYDnXKn6lg7StqlYBZGggTCuP6k0ap9Q4-53_Vo88Rw221'
    # Usando biblioteca Genius, verificar possibilidade de criar sua propria lib
    genius = Genius(token, excluded_terms=[
                    "Remix", "Live", "Acoustic", "Version", "Vevo", "Intro"], remove_section_headers=True)
    artist = genius.search_artist("Aurora", max_songs=30    , sort="title")
    print('Carregando musicas...')
    print(artist.songs)

    all_lyrics = []

    for song in artist.songs:
        lyrics = song.lyrics
        all_lyrics.extend(lyrics.split())

    return all_lyrics


# Técnicas de NLP
def convert_to_lower_case(words):
    return [word.lower() for word in words]


def remove_punctuation(words):
    translator = str.maketrans('', '', string.punctuation)
    return [word.translate(translator) for word in words]


def remove_stopwords(words):
    stop_words = set(stopwords.words('english'))

    all_lyrics_no_stopwords = []
    for word in words:
        tokens = word_tokenize(word)
        lyrics_no_stopwords = [
            word for word in tokens if word not in stop_words]
        all_lyrics_no_stopwords.append(' '.join(lyrics_no_stopwords))

    return all_lyrics_no_stopwords


def execute_lemmatization(words):
    lemmatizer = WordNetLemmatizer()

    all_lyrics_lemmatized = []
    for lyrics in words:
        tokens = word_tokenize(lyrics)
        lyrics_lemmatized = [lemmatizer.lemmatize(word) for word in tokens]
        all_lyrics_lemmatized.append(' '.join(lyrics_lemmatized))

    return all_lyrics_lemmatized


def show_cliques_report(cliques):
    # Contar o número de cliques de cada dimensão
    clique_counts = {}
    for clique in cliques:
        size = len(clique)
        if size not in clique_counts:
            clique_counts[size] = 0
        clique_counts[size] += 1

    # Ordenar as cliques pelo tamanho das dimensões em ordem decrescente
    sorted_cliques = sorted(clique_counts.items(),
                            key=lambda x: x[0], reverse=True)

    # Emitir o relatório no terminal
    for size, count in sorted_cliques:
        print(f"Foram encontradas {count} cliques de dimensão K{size}")


def load_louvain_cliques_graph(graph, cliques):
    # Criar um grafo com as cliques como nós
    louvain_clique_graph = load_louvain_cliques_graph_nodes(cliques)
    louvain_clique_graph = load_louvain_cliques_graph_edges(graph, louvain_clique_graph, cliques)

    return louvain_clique_graph


def load_color_map(clique_graph, partition):
    # Criar uma lista de cores
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple',
          'brown', 'pink', 'gray', 'olive', 'navy', 'teal', 'salmon',
          'indigo', 'gold', 'lime']

    # Criar um dicionário de cores para os clusters
    color_map = {node: colors[partition[node]]
                 for node in clique_graph.nodes()}

    return color_map


def apply_nlp_methods(all_lyrics):
    all_lyrics = convert_to_lower_case(all_lyrics)
    all_lyrics = remove_punctuation(all_lyrics)
    all_lyrics = remove_stopwords(all_lyrics)
    all_lyrics = list(filter(None, all_lyrics))
    all_lyrics = execute_lemmatization(all_lyrics)
    return all_lyrics

def load_clique_graph(cliques):
    # Criar um grafo vazio para representar o grafo de cliques
    clique_graph = nx.Graph()

    # Adicionar nós ao grafo, onde cada nó representa uma clique
    for i, clique in enumerate(cliques):
        clique_graph.add_node(i)

    # Adicionar as arestas entre os nós do grafo de cliques
    for i, clique1 in enumerate(cliques):
        for j, clique2 in enumerate(cliques):
            if i != j and any(node in clique2 for node in clique1):
                clique_graph.add_edge(i, j)
                
    return clique_graph

def plot_graph_laplacian_spectrum(graph):
    # Obtém a matriz Laplaciana do grafo
    laplacian_matrix = nx.laplacian_matrix(graph).toarray()

    # Calcula os autovalores da matriz Laplaciana
    eigenvalues = np.linalg.eigvals(laplacian_matrix)

    # Ordena os autovalores em ordem crescente
    eigenvalues.sort()

    # Plot dos autovalores
    plt.plot(eigenvalues)
    plt.xlabel('Índice')
    plt.ylabel('Autovalor')
    plt.savefig('spectrum.png')
    plt.close()

def export_cliques_report_cvs(cliques):
    # Criação do dataframe para o relatório
    cliques_report = pd.DataFrame(columns=['Dimensão', 'Clique'])

    # Preenchimento do dataframe com as informações das cliques
    for clique in cliques:
        dimensao = len(clique)
        cliques_report = cliques_report._append({'Dimensão': dimensao, 'Clique': ', '.join(clique)}, ignore_index=True)

    cliques_report = cliques_report.sort_values(by='Dimensão')

    # Exportar o dataframe para um arquivo CSV
    cliques_report.to_csv('cliques_report.csv', index=False)




all_lyrics = load_all_lyrics()
# Aplicar algumas técnicas usadas em NLP
all_lyrics = apply_nlp_methods(all_lyrics)
graph = load_correlation_graph(all_lyrics)

plot_graph_laplacian_spectrum(graph)

# Visualização do grafo
nx.draw(graph, with_labels=True, node_size=20, width=0.1, font_size=4)
plt.savefig('grafo.png', dpi=600)
plt.close()

# Cliques e algoritmo de Louvian
cliques = list(nx.find_cliques(graph))

show_cliques_report(cliques)
export_cliques_report_cvs(cliques)

print('graph:')
print(graph)

node_word_map = {i: word for i, word in enumerate(graph.nodes())}
# Criar um grafo com as cliques como nós
clique_graph_lovain = load_louvain_cliques_graph(graph, cliques)

print('clique_graph_lovain:')
print(clique_graph_lovain)

# Executar o algoritmo Louvain para clusterização
partition = community.best_partition(clique_graph_lovain)
color_map = load_color_map(clique_graph_lovain, partition)

pos = nx.spring_layout(clique_graph_lovain)
# Plotar os nós
nx.draw_networkx_nodes(clique_graph_lovain, pos, node_color=[
                       color_map[node] for node in clique_graph_lovain.nodes()], node_size=2)

# Plotar as arestas com largura máxima mais fina
nx.draw_networkx_edges(clique_graph_lovain, pos, width=0.05)
nx.draw_networkx_labels(clique_graph_lovain, pos, font_size=1)


show_louvain_partition_report(partition)
