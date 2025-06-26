import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score

# Ścieżki do danych
ratings_path = 'Movie_recommender/ml-100k/u.data'
movies_path = 'Movie_recommender/ml-100k/u.item'
users_path = 'Movie_recommender/ml-100k/u.user'

# Gatunki (kolejność alfabetyczna)
genres = ['unknown','Action','Adventure','Animation','Children','Comedy','Crime','Documentary',
          'Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi',
          'Thriller','War','Western']

# Wczytanie ocen
ratings = pd.read_csv(ratings_path, sep='\t', header=None, names=['user_id','movie_id','rating','timestamp'])

# Wczytanie danych o filmach
movies = pd.read_csv(movies_path, sep='|', header=None, encoding='latin-1',
                     names=['movie_id','title','release_date','video_release_date','IMDb_URL'] + genres)
movies = movies[['movie_id','title'] + genres]

# Wczytanie danych demograficznych
users = pd.read_csv(users_path, sep='|', header=None,
                    names=['user_id', 'age', 'gender', 'occupation', 'zip_code'])
users = users.drop(columns=['zip_code'])

# Zmienne kategoryczne użytkowników
users = pd.get_dummies(users, columns=['gender', 'occupation'])

# Podział danych na treningowe i testowe
train, test = train_test_split(ratings, test_size=0.2, random_state=42)

# Filtrowanie użytkowników w zbiorze testowym
test_users = test.groupby('user_id').filter(lambda x: len(x) >= 5)['user_id'].unique()

# Budowa macierzy ocen z danych treningowych
rating_matrix = train.pivot(index='user_id', columns='movie_id', values='rating')

# Tworzymy wektor profilu użytkownika (ważony ocenami)
def get_weighted_profile(user_id):
    user_ratings = train[(train.user_id == user_id) & (train.rating >= 4)]
    if user_ratings.empty:
        return None
    liked_movies = movies.set_index('movie_id').loc[user_ratings.movie_id]
    weights = user_ratings.rating.values.reshape(-1, 1)
    profile = np.average(liked_movies[genres].values, axis=0, weights=weights.flatten())
    return profile.reshape(1, -1)

def recommend_content_based(user_id, top_k=5):
    profile = get_weighted_profile(user_id)
    if profile is None:
        return []
    movie_vectors = movies.set_index('movie_id')[genres].values
    similarities = cosine_similarity(profile, movie_vectors).flatten()
    rated = set(train[train.user_id == user_id].movie_id)
    candidates = [(mid, score) for mid, score in zip(movies.movie_id, similarities) if mid not in rated]
    top = sorted(candidates, key=lambda x: x[1], reverse=True)[:top_k]
    return [movies.set_index('movie_id').loc[mid].title for mid, _ in top]

def recommend_user_user(user_id, top_k=5, neighbor_count=20):
    if user_id not in rating_matrix.index:
        return []
    target_vector = rating_matrix.loc[user_id]
    similarities = {}

    for other_id in rating_matrix.index:
        if other_id == user_id:
            continue
        other_vector = rating_matrix.loc[other_id]

        common = target_vector.notna() & other_vector.notna()
        if common.sum() < 2:
            continue

        ratings1 = target_vector[common].values
        ratings2 = other_vector[common].values

        if np.std(ratings1) == 0 or np.std(ratings2) == 0:          # Odchylenie standardowe
            continue

        corr = np.corrcoef(ratings1, ratings2)[0, 1]

        if np.isnan(corr) or corr <= 0:
            continue

        user_demo = users.set_index('user_id').loc[user_id].values
        other_demo = users.set_index('user_id').loc[other_id].values
        demo_sim = cosine_similarity([user_demo], [other_demo])[0, 0]

        similarities[other_id] = (corr + demo_sim) / 2

    if not similarities:
        return []

    neighbors = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:neighbor_count]
    neighbor_ids = [nid for nid, _ in neighbors]

    watched = set(train[train.user_id == user_id].movie_id)
    neighbor_ratings = train[(train.user_id.isin(neighbor_ids)) & (~train.movie_id.isin(watched)) & (train.rating >= 4)]
    if neighbor_ratings.empty:
        return []

    movie_scores = neighbor_ratings.groupby('movie_id')['rating'].mean()
    top = movie_scores.sort_values(ascending=False).head(top_k).index
    return [movies.set_index('movie_id').loc[mid].title for mid in top]

def recommend_hybrid(user_id, top_k=5):
    cb = recommend_content_based(user_id, top_k=50)
    cf = recommend_user_user(user_id, top_k=50)
    if not cb and not cf:
        return []

    hybrid_scores = {}
    for idx, title in enumerate(cb):
        hybrid_scores[title] = hybrid_scores.get(title, 0) + (50 - idx)
    for idx, title in enumerate(cf):
        hybrid_scores[title] = hybrid_scores.get(title, 0) + (50 - idx)

    top = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [title for title, _ in top]

def precision_at_k(recommended, relevant, k):
    if not recommended:
        return 0.0
    recommended_at_k = recommended[:k]
    relevant_set = set(relevant)
    return len([item for item in recommended_at_k if item in relevant_set]) / k

def jaccard_similarity(list1, list2):
    set1, set2 = set(list1), set(list2)
    intersection = set1 & set2
    union = set1 | set2
    return len(intersection) / len(union) if union else 0.0

def liked_movies(user_id):

    liked = ratings[(ratings['user_id'] == user_id) & (ratings['rating'] >= 4)]
    liked_with_genres = (liked.merge(movies, on='movie_id')).sort_values('rating', ascending=False)
    return liked_with_genres[['title', 'rating']]

if __name__ == "__main__":
    try:
        user_id = int(input("Podaj ID użytkownika (1-943): "))
    except ValueError:
        print("Błędne ID użytkownika")
        exit()

    print(liked_movies(user_id))      

    print(f"\nTop 5 rekomendacji modelu content-based dla użytkownika numer {user_id}:")
    cb_preds = recommend_content_based(user_id, top_k=5)
    for idx, title in enumerate(cb_preds, 1):
        print(f"{idx}. {title}")

    print(f"\nTop 5 rekomendacji modelu collaborative dla użytkownika numer {user_id}:")
    cf_preds = recommend_user_user(user_id, top_k=5)
    for idx, title in enumerate(cf_preds, 1):
        print(f"{idx}. {title}")

    print(f"\nTop 5 rekomendacji modelu hybrydowgo dla użytkownika numer {user_id}:")
    hy_preds = recommend_hybrid(user_id, top_k=5)
    for idx, title in enumerate(hy_preds, 1):
        print(f"{idx}. {title}")

    print("\nPorównanie rekomendacji:")
    for i in range(5):
        cb = cb_preds[i] if i < len(cb_preds) else "-"
        cf = cf_preds[i] if i < len(cf_preds) else "-"
        hy = hy_preds[i] if i < len(hy_preds) else "-"
        print(f"{i + 1}. {cb:<40} | {cf:<40} | {hy}")

    if user_id in test_users:
        relevant_movies = test[(test.user_id == user_id) & (test.rating >= 4)].movie_id
        relevant_titles = set(movies.set_index('movie_id').loc[relevant_movies]['title'])

        cb_prec = precision_at_k(cb_preds, relevant_titles, k=5)
        cf_prec = precision_at_k(cf_preds, relevant_titles, k=5)
        hy_prec = precision_at_k(hy_preds, relevant_titles, k=5)
        jaccard_cb_cf = jaccard_similarity(cb_preds, cf_preds)
        jaccard_cb_hy = jaccard_similarity(cb_preds, hy_preds)
        jaccard_cf_hy = jaccard_similarity(cf_preds, hy_preds)

        print(f"\nPrecyzja content-based: {cb_prec:.2f}")
        print(f"Precyzja collaborative: {cf_prec:.2f}")
        print(f"Precyzja hybrydowy: {hy_prec:.2f}")
        print(f"Zbieżność CB-CF : {jaccard_cb_cf:.2f}")
        print(f"Zbieżność CB-HY : {jaccard_cb_hy:.2f}")
        print(f"Zbieżność CF-HY : {jaccard_cf_hy:.2f}")
    else:
        print("\nBrak wystarczającej liczby ocen w zbiorze testowym do ewaluacji.")
