import networkx as nx
import matplotlib.pyplot as plt
from lyricsgenius import Genius
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

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


def load_all_lyrics():
    # Acesso a API do Genius
    token = 'HpJ0pnH7fP-sc9GodGLYDnXKn6lg7StqlYBZGggTCuP6k0ap9Q4-53_Vo88Rw221'
    # Usando biblioteca Genius, verificar possibilidade de criar sua propria lib
    genius = Genius(token)
    artist = genius.search_artist("Aurora", max_songs=5, sort="title")
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
        lyrics_no_stopwords = [word for word in tokens if word not in stop_words]
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

all_lyrics = load_all_lyrics()

# Aplicar algumas técnicas usadas em NLP
all_lyrics = convert_to_lower_case(all_lyrics)
all_lyrics = remove_punctuation(all_lyrics)
all_lyrics = remove_stopwords(all_lyrics)
all_lyrics = list(filter(None, all_lyrics))
all_lyrics = execute_lemmatization(all_lyrics)

graph = load_correlation_graph(all_lyrics)

# Visualização do grafo
nx.draw(graph, with_labels=True, node_size=20, width=0.1, font_size=4)
plt.savefig('grafo.png', dpi=600)
