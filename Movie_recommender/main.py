import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

ratings_path = 'ml-100k/u.data'
movies_path = 'ml-100k/u.item'

# Wczytanie ocen
ratings = pd.read_csv(ratings_path, sep='\t', header=None, names=['user_id','movie_id','rating','timestamp'])

# Lista gatunków z pliku u.genre (kolejność alfabetyczna)
genres = ['unknown','Action','Adventure','Animation','Children','Comedy','Crime','Documentary',
          'Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi',
          'Thriller','War','Western']

# Wczytanie informacji o filmach
movies = pd.read_csv(movies_path, sep='|', header=None, names=['movie_id','title','release_date','video_release_date','IMDb_URL'] + genres, encoding='latin-1')

# Pozbywamy się niepotrzebnych kolumn (można zachować tylko ID, tytuł i gatunki)
movies = movies[['movie_id','title'] + genres]

print(f"Liczba ocen: {len(ratings)}, liczba filmów: {len(movies)}")


def recommend_content_based(user_id, top_k=5):
    # Filtrujemy filmy, które użytkownik ocenił ≥ 4
    liked = ratings[(ratings.user_id == user_id) & (ratings.rating >= 4)]
    if liked.empty:
        return []  # brak filmów lubianych, nie można zbudować profilu

    # Tworzymy macierz wektorów cech filmów lubianych dla danego użytkownika
    liked_vectors = movies.set_index('movie_id').loc[liked.movie_id][genres].values
    # Profil użytkownika: średni wektor cech (każdy wymiar to średnia z ocenionych filmów)
    user_profile = np.mean(liked_vectors, axis=0).reshape(1, -1)

    # Macierz wektorów cech wszystkich filmów
    movie_vectors = movies.set_index('movie_id')[genres].values
    # Obliczamy podobieństwo kosinusowe między profilem, a wszystkimi filmami
    similarities = cosine_similarity(user_profile, movie_vectors).flatten()

    # Usuwamy filmy już ocenione przez użytkownika z rekomendacji
    rated_movies = set(ratings[ratings.user_id == user_id].movie_id)
    # Para (movie_id, similarity) dla nieocenionych filmów
    candidates = [(movie_id, score)
                  for movie_id, score in zip(movies.movie_id, similarities)
                  if movie_id not in rated_movies]

    # Sortujemy po podobieństwie (malejąco) i zwracamy top k
    top_movies = sorted(candidates, key=lambda x: x[1], reverse=True)[:top_k]
    return [movies.set_index('movie_id').loc[movie_id]['title'] for movie_id, _ in top_movies]


def recommend_user_user(user_id, top_k=5, neighbor_count=5):
    # Tworzymy macierz użytkownicy-filmy
    rating_matrix = ratings.pivot(index='user_id', columns='movie_id', values='rating')         # wiersze = użytkownicy / kolumny = filmy / wartości = oceny

    # Obliczamy macierz korelacji użytkowników (Pearson)
    user_corr = rating_matrix.T.corr(method='pearson')

    # Seria podobieństwa danego użytkownika do innych
    sim_series = user_corr[user_id].drop(labels=[user_id]).dropna()
    # Wybieramy sąsiadów o najwyższej dodatniej korelacji
    sim_series = sim_series[sim_series > 0]  # ignorujemy ujemne korelacje
    neighbors = sim_series.sort_values(ascending=False).head(neighbor_count).index.tolist()

    # Filtrujemy filmy ocenione wysoko (>=4) przez sąsiadów, których nie widział target
    user_movies = set(ratings[ratings.user_id == user_id].movie_id)
    neighbor_ratings = ratings[
        (ratings.user_id.isin(neighbors)) & (~ratings.movie_id.isin(user_movies)) & (ratings.rating >= 4)]          #za pamięci: 1 warunek - oceny najblizszych sasiadow / 2 warunek - filmy ktorych nie ogladał
    if neighbor_ratings.empty:
        return []

    # Grupujemy kandydatów i obliczamy średnią ocenę od sąsiadów
    movie_scores = neighbor_ratings.groupby('movie_id')['rating'].mean()
    # Sortujemy po średniej (malejąco) i bierzemy top k
    top_movies = movie_scores.sort_values(ascending=False).head(top_k).index
    return [movies.set_index('movie_id').loc[movie_id]['title'] for movie_id in top_movies]


if __name__ == "__main__":
    try:
        user_id = int(input("Podaj ID użytkownika (1-943): "))
    except ValueError:
        print("Błędne ID użytkownika")
        exit()

    print(f"\nTop 5 rekomendacji *content-based* dla użytkownika {user_id}:")
    content_recs = recommend_content_based(user_id, top_k=5)
    for idx, title in enumerate(content_recs, 1):
        print(f"{idx}. {title}")

    print(f"\nTop 5 rekomendacji *collaborative* dla użytkownika {user_id}:")
    cf_recs = recommend_user_user(user_id, top_k=5)
    for idx, title in enumerate(cf_recs, 1):
        print(f"{idx}. {title}")

    # Krótkie porównanie wyników (np. wypisanie obok siebie)
    print("\nPorównanie rekomendacji:")
    for i in range(max(len(content_recs), len(cf_recs))):
        cb = content_recs[i] if i < len(content_recs) else "-"
        cf = cf_recs[i] if i < len(cf_recs) else "-"
        print(f"{i + 1}. {cb:<50} | {cf}")