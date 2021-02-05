def findbyname(album_name, data):
    new_dict={}
    for album in data:
        if album['name']==album_name:
            return album

def findbyrank(album_rank, data):
    new_dict={}
    for album in data:
        if album['number'] == album_rank:
            return album

def findbyyear(album_year, data):
    all_albums = []
    for album in data:
        new_dict={}
        if album['year']==album_year:
            for k,v in album.items():
                new_dict[k] = v
            all_albums.append(new_dict)
    return all_albums

def findbyyears(start_year, end_year, data):
    all_albums = []
    for album in data:
        new_dict = {}
        if (int(album['year']) >= int(start_year) and int(album['year']) <= int(end_year)):
            for k,v in album.items():
                new_dict[k] = v
            all_albums.append(new_dict)
    return all_albums

def findbyranks(start_rank, end_rank, data):
    all_albums = []
    for album in data:
        new_dict = {}
        if int(album['number']) >= int(start_rank) and int(album['number']) <= int(end_rank):
            for k,v in album.items():
                new_dict[k] = v
            all_albums.append(new_dict)
    return all_albums

def all_titles(data):
    all_titles_list = []
    [all_titles_list.append(album['name']) for album in data]
    return all_titles_list

def all_artists(data):
    all_artists_list = []
    [all_artists_list.append(album['artist']) for album in data]
    return all_artists_list

def most_popular_artist(our_data):
    
    test_list = all_artists(data)
    top_artists = []
    artists_count = {}
    for artist in test_list:
        artists_count[artist] = artists_count.get(artist, 0)+1
    max_mentions = max(artists_count.values())
    max_mentions
    counts = list(artists_count.items())
    for artist in counts:
        if artist[1] == max_mentions:
            top_artists.append(artist[0])
    return top_artists

def most_popular_words(data):

    test_list = all_titles(data)
    top_words=[]
    words_count = {}
    for title in test_list:
        words = title.split()
        for word in words:
            words_count[word.upper()] = words_count.get(word.upper(), 0)+1
    counts = list(words_count.items())
    all_top_words = sorted(counts, key = lambda x: x[1], reverse=True)[:25]#for word in counts:
    all_top_words
    max_words = max(words_count.values())
    for word in all_top_words:
        if word[1] == max_words:
            top_words.append(word[0])
    return top_words

def hist_of_albums_decade(data):
    x_years = [str(x) for x in list(range(1950,2020))]
    y_vals = []
    for year in x_years:
        num_by_year = len(functions.findbyyear(year, list_all))
        for x in list(range(0,num_by_year)):
            y_vals.append(int(year))
    fig, ax = plt.subplots()
    ax.hist(y_vals, bins=5)
    ax.set_title("Albums By Decade")
    ax.set_xlabel("Decade")
    ax.set_ylabel("Number of Albums")

def hist_by_genre(data):
    genre_list = []
    for album in data:
        album_genre = album['genre']
        for genre in album_genre.split(','):
            genre = genre.replace(" ","")
            genre = genre.replace("&","")
            genre_list.append(genre)
    y_vals = genre_list
    fig, ax = plt.subplots()
    ax.hist(y_vals)
    ax.set_title("Genre Frequency List")
    ax.set_xlabel("Genre")
    ax.set_ylabel("Frequency Count")

def albumsWithMostTopSongs(top_songs_data, track_data):

    ranks_dict = {}
    for song in top_songs_data:
        for tracks in track_data:
            if song['name'] in tracks['tracks']:
                if tracks['album'] in ranks_dict.keys():
                    ranks_dict[tracks['album']] = ranks_dict.get(tracks['album'], 0) + 1
                else:
                    ranks_dict[tracks['album']] = 1
    top_album = sorted(ranks_dict.items(), key=lambda x: x[1], reverse=True)
    for tracks in track_data:
        if top_album[0][0] == tracks['album']:
            artist_album = tracks['artist']
    return [artist_album, top_album[0][0], top_album[0][1]]

def albumsWithTopSongs(top_songs_data, tracks_data):

    album_list = []
    for song in top_songs_data:
        for tracks in tracks_data:
            if song['name'] in tracks['tracks']:
                album_list.append(tracks['album'])
    return album_list

def songsThatAreOnTopAlbums(top_songs_data, tracks_data):

    songs_list = []
    album_list = albumsWithTopSongs(top_songs_data, tracks_data)
    for album in album_list:
        for track in tracks_data:
            if album == track['album']:
                for each_song in track['tracks']:
                    songs_list.append(each_song)
        clean_list = set(songs_list)
    return clean_list

def top10AlbumsByTopSongs(top_songs_data, tracks_data):
    ranks_dict = {}
    for song in top_songs_data:
        for tracks in tracks_data:
            if song['name'] in tracks['tracks']:
                if tracks['album'] in ranks_dict.keys():
                    ranks_dict[tracks['album']] = ranks_dict.get(tracks['album'], 0) + 1
                else:
                    ranks_dict[tracks['album']] = 1
    top_albums = sorted(ranks_dict.items(), key=lambda x: x[1], reverse=True)[:10]
    y_vals = []
    for i in top_albums:
        for x in list(range(0,i[1])):
            y_vals.append(i[0])
    fig, ax = plt.subplots()
    ax.hist(y_vals)
    ax.set_title("Top 10 Albums by Top Songs")
    ax.set_xlabel("Album")
    ax.set_ylabel("Frequency")


