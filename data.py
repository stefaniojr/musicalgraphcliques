import os
from lyricsgenius import Genius

def load_all_lyrics(artist, qty_musics):
    # Acesso a API do Genius
    token = 'HpJ0pnH7fP-sc9GodGLYDnXKn6lg7StqlYBZGggTCuP6k0ap9Q4-53_Vo88Rw221'
    # Usando biblioteca Genius, verificar possibilidade de criar sua própria lib
    genius = Genius(token, excluded_terms=[
                    "Remix", "Live", "Acoustic", "Version", "Vevo", "Intro", "Tour", "Speech", "Mix", "Demo", "Unreleased"], remove_section_headers=True)

    all_lyrics = []

    for artist_name in artist:
        artist_obj = genius.search_artist(artist_name, sort='popularity', max_songs=qty_musics)
        for song in artist_obj.songs:
            lyrics = song.lyrics
            all_lyrics.extend(lyrics.split())

    return all_lyrics


def export_lyrics_to_txt(artist, all_lyrics):
    folder_name = 'artists_data'
    
    for artist_name in artist:
        file_name = f'{artist_name.lower().replace(" ", "_")}.txt'
        folder_path = os.path.join(os.getcwd(), folder_name)

        # Cria a pasta se ainda não existir
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        file_path = os.path.join(folder_path, file_name)

        with open(file_path, 'w', encoding='utf-8') as file:
            file.write('\n'.join(all_lyrics))

        print(f"Letras do artista '{artist_name}' exportadas para '{file_path}'.")


# Exemplo de uso
artists = ['Demi Lovato']
qty_musics = 25

for artist in artists:
    all_lyrics = load_all_lyrics([artist], qty_musics)
    export_lyrics_to_txt([artist], all_lyrics)