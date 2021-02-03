def findbyname(album_name, data):
    new_dict={}
    for album in data:
        if album['album']==album_name:
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
    [all_titles_list.append(album['album']) for album in data]
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

