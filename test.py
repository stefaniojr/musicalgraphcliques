import requests

# Defina sua chave de acesso à API do Genius
genius_token = 'HpJ0pnH7fP-sc9GodGLYDnXKn6lg7StqlYBZGggTCuP6k0ap9Q4-53_Vo88Rw221'
base_url = 'https://api.genius.com'
# URL da rota de pesquisa para obter o ID do artista
search_url = 'https://api.genius.com/search'
artist_name = 'Taylor Swift'

def find_album_api_paths(data):
    results = []
    if isinstance(data, dict):
        for key, value in data.items():
            if key == "api_path" and value.startswith("/albums/"):
                results.append(value)
            elif isinstance(value, (dict, list)):
                results.extend(find_album_api_paths(value))
    elif isinstance(data, list):
        for item in data:
            results.extend(find_album_api_paths(item))
    return results

def find_songs_api_paths(data):
    results = []
    if isinstance(data, dict):
        for key, value in data.items():
            if key == "api_path" and value.startswith("/songs/"):
                results.append(value)
            elif isinstance(value, (dict, list)):
                results.extend(find_songs_api_paths(value))
    elif isinstance(data, list):
        for item in data:
            results.extend(find_songs_api_paths(item))
    return results

def get_album_info(album_path):
    headers = {'Authorization': 'Bearer ' + genius_token}

    # Faça a solicitação GET para obter informações do álbum
    response = requests.get(base_url + album_path, headers=headers)

    # Verifique se a solicitação foi bem-sucedida (código de status 200)
    if response.status_code == 200:
        album_info = response.json()
        return album_info
    else:
        print('Erro na solicitação:', response.status_code)
        return None
    
def get_song_info(song_path):
    headers = {'Authorization': 'Bearer ' + genius_token}

    # Faça a solicitação GET para obter informações do álbum
    response = requests.get(base_url + song_path, headers=headers)

    # Verifique se a solicitação foi bem-sucedida (código de status 200)
    if response.status_code == 200:
        song_info = response.json()
        return song_info
    else:
        print('Erro na solicitação:', response.status_code)
        return None

# Parâmetros da requisição de pesquisa
params = {
    'access_token': genius_token,
    'q': artist_name,
    'per_page': 1
}

# Fazer a requisição de pesquisa para obter o ID do artista
response = requests.get(search_url, params=params)

# Verificar se a requisição foi bem-sucedida
if response.status_code == 200:
    search_data = response.json()

    if 'hits' in search_data['response']:
        hits = search_data['response']['hits']

        # Verificar se há algum hit retornado na pesquisa
        if len(hits) > 0:
            artist = hits[0]['result']['primary_artist']
            artist_id = artist['id']
            artist_name = artist['name']

            print(f'Artista encontrado {artist_name} com id {artist_id}')
            
            artist_url = f'https://api.genius.com/artists/{artist_id}'

            # Parâmetros da requisição para obter os álbuns
            params = {
                'access_token': genius_token
            }

            # Fazer a requisição para obter os álbuns do artista
            response = requests.get(artist_url, params=params)
            

            if response.status_code == 200:
                album_paths =  list(set(find_album_api_paths(response.json())))

                for album in album_paths:
                    album_info = get_album_info(album)
                    if album_info:
                        album_title = album_info['response']['album']['name']
                        print('album info', album_info)
                        if 'Version' not in album_title and 'Acoustic' not in album_title:
                            print('\n')
                            print(f'Obtendo musicas do album {album_title}:')
                            song_paths =  list(set(find_songs_api_paths(album_info)))

                            for song in song_paths:
                                song_info = get_song_info(song)
                                if song_info:
                                    print('- ', song_info['response']['song']['title'])

                    else:
                        print('Não foi possível obter informações do álbum', album)
            
            else:
                print(f'Falha ao obter informações do artista {artist_name}.')

        else:
            print(f'Nenhum artista encontrado com o nome {artist_name}.')
        

    else:
        print('Nenhum resultado de pesquisa encontrado.')

else:
    print(f'Falha na requisição de pesquisa do artista {artist_name}.')