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
import os
from PIL import Image
import shutil


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

def load_correlation_graph(words, context):
    graph = nx.Graph()

    # Percorre todas as palavras
    for i in range(len(words)):
        word = words[i]

        # Adicione o nó ao grafo se ainda não existir
        if word not in graph:
            graph.add_node(word)

        # Verifique as palavras vizinhas dentro de um contexto definido
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


def load_all_lyrics(artist, qty_musics):
    # Acesso a API do Genius
    token = 'HpJ0pnH7fP-sc9GodGLYDnXKn6lg7StqlYBZGggTCuP6k0ap9Q4-53_Vo88Rw221'
    # Usando biblioteca Genius, verificar possibilidade de criar sua propria lib
    genius = Genius(token, excluded_terms=[
                    "Remix", "Live", "Acoustic", "Version", "Vevo", "Intro", "Tour", "Speech", "Mix", "Demo", "Unreleased"], remove_section_headers=True)
    artist = genius.search_artist(artist, max_songs=qty_musics)
    print(artist.songs)

    # Busque os álbuns do artista
    albums = artist.search_albums()


    # Busque todas as músicas do artista
    musicas = artist.songs

    # Filtrar apenas as músicas de versões padrão de álbuns de estúdio
    musicas_filtradas = []
    for musica in musicas:
        if musica.album is not None and musica.album.is_standard and musica.album.album_type == 'Album':
            musicas_filtradas.append(musica)


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

def plot_graph_laplacian_spectrum(graph, folder_path, artist, context):
    # Obtém a matriz Laplaciana do grafo
    laplacian_matrix = nx.laplacian_matrix(graph).toarray()

    # Calcula os autovalores da matriz Laplaciana
    eigenvalues = np.linalg.eigvals(laplacian_matrix)

    # Ordena os autovalores em ordem crescente
    eigenvalues.sort()

    # Plot dos autovalores
    plt.plot(eigenvalues)
    plt.title('Espectro da matriz laplaciana do grafo')
    plt.suptitle(f'Artista: {artist} | Contexto: {context}')
    plt.xlabel('Índice')
    plt.ylabel('Autovalor')
    plt.savefig(os.path.join(folder_path, 'spectrum.png'))
    plt.close()

def export_cliques_report(cliques, folder_path, artist, context):
    # Criação do dataframe para o relatório
    cliques_report = pd.DataFrame(columns=['Dimensão', 'Clique'])

    # Preenchimento do dataframe com as informações das cliques
    for clique in cliques:
        dimensao = len(clique)
        cliques_report = cliques_report._append({'Dimensão': dimensao, 'Clique': ', '.join(clique)}, ignore_index=True)

    cliques_report = cliques_report.sort_values(by='Dimensão')

    # Exportar o dataframe para um arquivo CSV
    cliques_report.to_csv(os.path.join(folder_path, 'cliques_report.csv'), index=False)

    # Plotar histograma das dimensões das cliques
    plt.hist(cliques_report['Dimensão'], bins=range(min(cliques_report['Dimensão']), max(cliques_report['Dimensão']) + 2, 1), edgecolor='black')
    plt.xlabel('Dimensão')
    plt.ylabel('Frequência')
    plt.title('Histograma das Dimensões das Cliques')
    plt.suptitle(f'Artista: {artist} | Contexto: {context}')
    plt.grid(True)
    plt.savefig(os.path.join(folder_path, f'histograma_cliques.png'))  # Salvar o histograma como uma imagem
    plt.close()  # Fechar o plot do histograma

def execute(artist, all_lyrics, context):
    graph = load_correlation_graph(all_lyrics, context)

    artist_name_folder = artist.lower().replace(" ", "_")
    folder_path = f"reports/{artist_name_folder}/context{context}"
    os.makedirs(folder_path, exist_ok=True)

    # Visualização do grafo
    nx.draw(graph, with_labels=True, node_size=20, width=0.1, font_size=4)
    plt.savefig(os.path.join(folder_path, 'grafo.png'), dpi=600)
    plt.close()

    plot_graph_laplacian_spectrum(graph, folder_path, artist, context)

    # Cliques
    cliques = list(nx.find_cliques(graph))
    show_cliques_report(cliques)
    export_cliques_report(cliques, folder_path, artist, context)

    # Algoritmo de Louvain
    #partition = community.best_partition(graph)

    #Imprimir resultados
    #show_louvain_partition_report(partition)

def delete_folder(artist):
    
    artist_name_folder = artist.lower().replace(" ", "_")
    folder_path = f'reports/{artist_name_folder}'
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        shutil.rmtree(folder_path)

def export_geral_histograma_reports(artist, context):
    images = []  # Lista para armazenar as imagens a serem combinadas

    artist_name_folder = artist.lower().replace(" ", "_")
    # Itera sobre as subpastas e encontra o arquivo histograma_cliques.png em cada uma
    for i in range(1, context + 1):
        image_path = os.path.join('reports', f'{artist_name_folder}', f'context{i}', f'histograma_cliques.png')
        if os.path.exists(image_path):
            image = Image.open(image_path)
            images.append(image)

    # Calcula o número de linhas e colunas
    num_histograms = len(images)
    num_columns = 2
    num_lines = (num_histograms + 1) // num_columns

    # Calcula a altura da imagem final
    image_width, image_height = images[0].size
    final_image_height = num_lines * image_height

    # Cria uma nova imagem em branco com as dimensões adequadas
    final_image = Image.new('RGB', (num_columns * images[0].width, final_image_height), color='white')


    # Combina as imagens na imagem final
    for i, image in enumerate(images):
        x = (i % num_columns) * image_width
        y = (i // num_columns) * image_height
        final_image.paste(image, (x, y))

        # Salva a imagem final
        final_image.save(os.path.join('reports', f'{artist_name_folder}', f'report_histogram.png'))

def export_geral_spectrum_reports(artist, context):
    images = []  # Lista para armazenar as imagens a serem combinadas

    artist_name_folder = artist.lower().replace(" ", "_")
    # Itera sobre as subpastas e encontra o arquivo histograma_cliques.png em cada uma
    for i in range(1, context + 1):
        image_path = os.path.join('reports', f'{artist_name_folder}', f'context{i}', f'spectrum.png')
        if os.path.exists(image_path):
            image = Image.open(image_path)
            images.append(image)

    # Calcula o número de linhas e colunas
    num_spectrums = len(images)
    num_columns = 2
    num_lines = (num_spectrums + 1) // num_columns

    # Calcula a altura da imagem final
    image_width, image_height = images[0].size
    final_image_height = num_lines * image_height

    # Cria uma nova imagem em branco com as dimensões adequadas
    final_image = Image.new('RGB', (num_columns * images[0].width, final_image_height), color='white')

    # Combina as imagens na imagem final
    for i, image in enumerate(images):
        x = (i % num_columns) * image_width
        y = (i // num_columns) * image_height
        final_image.paste(image, (x, y))

        # Salva a imagem final
        final_image.save(os.path.join('reports', f'{artist_name_folder}', f'report_spectrum.png'))

context = 10
qty_musics = 20
artists = ['Sia']

for artist in artists:
    delete_folder(artist)
    print(f'Analisando {artist}...')
    all_lyrics = []
    all_lyrics = load_all_lyrics(artist, qty_musics)
    # Aplicar algumas técnicas usadas em NLP
    all_lyrics = apply_nlp_methods(all_lyrics)
    
    for i in range(1, context + 1):
        print(f'Coletando dados para o contexto {i}...')
        execute(artist, all_lyrics, i)
    
    export_geral_histograma_reports(artist, context)
    export_geral_spectrum_reports(artist, context)