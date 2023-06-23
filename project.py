import networkx as nx
import matplotlib.pyplot as plt

import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import os
from PIL import Image
import shutil
import csv
from nltk.sentiment import SentimentIntensityAnalyzer

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


def apply_nlp_methods(all_lyrics):
    all_lyrics = convert_to_lower_case(all_lyrics)
    all_lyrics = remove_punctuation(all_lyrics)
    all_lyrics = remove_stopwords(all_lyrics)
    all_lyrics = list(filter(None, all_lyrics))
    all_lyrics = execute_lemmatization(all_lyrics)
    return all_lyrics


def export_cliques_report(cliques, folder_path, artist, context):
    # Criação do dataframe para o relatório
    cliques_report = pd.DataFrame(columns=['Dimensão', 'Clique'])

    # Preenchimento do dataframe com as informações das cliques
    for clique in cliques:
        dimensao = len(clique)
        cliques_report = cliques_report._append(
            {'Dimensão': dimensao, 'Clique': ', '.join(clique)}, ignore_index=True)

    cliques_report = cliques_report.sort_values(by='Dimensão')

    # Exportar o dataframe para um arquivo CSV
    cliques_report.to_csv(os.path.join(
        folder_path, 'cliques_report.csv'), index=False)

    # Plotar histograma das dimensões das cliques
    plt.hist(cliques_report['Dimensão'], bins=range(min(cliques_report['Dimensão']), max(
        cliques_report['Dimensão']) + 2, 1), edgecolor='black')
    plt.xlabel('Dimensão')
    plt.ylabel('Frequência')
    plt.title('Histograma das Dimensões das Cliques')
    plt.suptitle(f'Artista: {artist} | Contexto: {context}')
    plt.grid(True)
    # Salvar o histograma como uma imagem
    plt.savefig(os.path.join(folder_path, f'histograma_cliques.png'))
    plt.close()  # Fechar o plot do histograma


def export_cliques_emotion_report(folder_path):
    # Inicializar o classificador de emoções
    sid = SentimentIntensityAnalyzer()

    # Abrir arquivo de entrada para leitura
    with open(os.path.join(folder_path, 'cliques_report.csv'), 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)

        # Inicializar o analisador de sentimentos
        sid = SentimentIntensityAnalyzer()

        # Abrir arquivo de saída para escrita
        with open(os.path.join(folder_path, 'cliques_emotion_report.csv'), 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            # Escrever o cabeçalho no arquivo de saída
            writer.writerow(['Clique', 'Emoção'])

            # Analisar cada linha do arquivo CSV de entrada
            for row in reader:
                clique = row['Clique']

                # Realizar a análise de sentimento para cada clique
                sentiment_scores = sid.polarity_scores(clique)
                # Score de sentimento geral
                sentiment_score = sentiment_scores['compound']

                # Atribuir emoção com base no valor de sentimento
                emotion = sentiment_score

                # Escrever o clique e a emoção atribuída no arquivo de saída
                writer.writerow([clique, emotion])

    print("Atribuição de emoções concluída. O relatório foi salvo.")

def plot_emotions(folder_path):
    # Carregar o arquivo de entrada como um DataFrame
    df = pd.read_csv(os.path.join(folder_path, 'cliques_emotion_report.csv'))

    # Obter a quantidade de itens em cada clique
    df['Cliques'] = df['Clique'].apply(lambda x: len(x.split(',')))

    # Gerar o gráfico de dispersão
    plt.scatter(df['Cliques'], df['Emoção'])
    plt.xlabel('Cliques')
    plt.ylabel('Emoção')
    plt.title('Emoções por Dimensão da Clique')
    plt.savefig(os.path.join(folder_path, f'emotions_graph.png'))
    plt.close() 

def plot_emotion_histogram(folder_path):
    # Carregar o arquivo de entrada como um DataFrame
    df = pd.read_csv(os.path.join(folder_path, 'cliques_emotion_report.csv'))

    # Definir os limites para cada região emocional
    limites = [-1, -0.5, -0.0001, 0.0001, 0.5, 1]

    # Agrupar as emoções em categorias
    categorias = ['Muito Negativas', 'Negativas', 'Neutras', 'Positivas', 'Muito Positivas']

    df['Categoria'] = pd.cut(df['Emoção'], bins=limites, labels=categorias, right=False)

    # Contar a frequência de cada região emocional
    frequencias = df['Categoria'].value_counts()

    # Gerar o histograma de frequência
    labels = frequencias.index
    frequencies = frequencias.values

    plt.bar(labels, frequencies)
    plt.xlabel('Região Emocional')
    plt.ylabel('Frequência')
    plt.title('Histograma de Frequência das Regiões Emocionais')
    plt.savefig(os.path.join(folder_path, f'emotion_histograma.png'))
    plt.close()  # Fechar o plot do histograma


def execute(artist, all_lyrics, context):
    graph = load_correlation_graph(all_lyrics, context)

    folder_path = f"reports/{artist}/context{context}"
    os.makedirs(folder_path, exist_ok=True)

    # Visualização do grafo
    nx.draw(graph, node_size=1, width=0.03, font_size=4)
    plt.savefig(os.path.join(folder_path, 'grafo.png'), dpi=600)
    plt.close()

    # Cliques
    cliques = list(nx.find_cliques(graph))
    show_cliques_report(cliques)
    export_cliques_report(cliques, folder_path, artist, context)
    export_cliques_emotion_report(folder_path)
    plot_emotions(folder_path)
    plot_emotion_histogram(folder_path)


def delete_folder(artist):
    folder_path = f'reports/{artist}'
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        shutil.rmtree(folder_path)


def export_geral_histograma_reports(artist, context):
    images = []  # Lista para armazenar as imagens a serem combinadas

    # Itera sobre as subpastas e encontra o arquivo histograma_cliques.png em cada uma
    for i in range(1, context + 1):
        image_path = os.path.join(
            'reports', f'{artist}', f'context{i}', f'histograma_cliques.png')
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
    final_image = Image.new(
        'RGB', (num_columns * images[0].width, final_image_height), color='white')

    # Combina as imagens na imagem final
    for i, image in enumerate(images):
        x = (i % num_columns) * image_width
        y = (i // num_columns) * image_height
        final_image.paste(image, (x, y))

        # Salva a imagem final
        final_image.save(os.path.join(
            'reports', f'{artist}', f'report_histogram.png'))
        
def export_geral_emotion_histograma_reports(artist, context):
    images = [] 

    for i in range(1, context + 1):
        image_path = os.path.join(
            'reports', f'{artist}', f'context{i}', f'emotion_histograma.png')
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
    final_image = Image.new(
        'RGB', (num_columns * images[0].width, final_image_height), color='white')

    # Combina as imagens na imagem final
    for i, image in enumerate(images):
        x = (i % num_columns) * image_width
        y = (i // num_columns) * image_height
        final_image.paste(image, (x, y))

        # Salva a imagem final
        final_image.save(os.path.join(
            'reports', f'{artist}', f'report_emotion_histogram.png'))


context = 7
folder = "artists_data"
artists = []

for file in os.listdir(folder):
    if file.endswith(".txt"):
        artist_name = file.split(".txt")[0]
        file_path = os.path.join(folder, file)
        with open(file_path, "r", encoding='utf-8') as artist_file:
            lines = artist_file.read().splitlines()
            artist_data = {"name": artist_name, "lyrics": lines}
            artists.append(artist_data)

for artist in artists:
    delete_folder(artist['name'])
    name = artist['name']
    print(f'Analisando {name}...')
    # Aplicar algumas técnicas usadas em NLP
    all_lyrics = apply_nlp_methods(artist['lyrics'])

    for i in range(3, context + 1):
        print(f'Coletando dados para o contexto {i}...')
        execute(artist['name'], all_lyrics, i)

    export_geral_histograma_reports(artist['name'].title(), context)
    export_geral_emotion_histograma_reports(artist['name'].title(), context)
