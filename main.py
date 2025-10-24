import pandas as pd
import numpy as np


ANIME_CSV = "anime.csv"
RATINGS_CSV = "rating.csv"

ACTION = "similar_to"      
SIMILAR_TO_TITLE = "Naruto"    
TARGET_USER_ID = 1             
TOP_K = 10                     
MIN_RATINGS_PER_ITEM = 5      

def load_data(anime_path: str, ratings_path: str):
    """Carga CSVs del dataset de Kaggle y limpia ratings -1."""
    anime_cols = ['anime_id', 'name', 'genre', 'type', 'episodes', 'rating', 'members']
    rating_cols = ['user_id', 'anime_id', 'rating']

    anime = pd.read_csv(anime_path, names=anime_cols, header=0)
    ratings = pd.read_csv(ratings_path, names=rating_cols, header=0)


    ratings = ratings[ratings['rating'] != -1].copy()

    return anime, ratings


def build_user_item_matrix(anime: pd.DataFrame,
                           ratings: pd.DataFrame,
                           min_ratings_per_item: int = 5):
    
    if min_ratings_per_item > 1:
        counts = ratings['anime_id'].value_counts()
        keep = counts[counts >= min_ratings_per_item].index
        ratings = ratings[ratings['anime_id'].isin(keep)].copy()

    ui = ratings.pivot_table(index='user_id',
                             columns='anime_id',
                             values='rating',
                             aggfunc='mean',
                             fill_value=0)

    M = ui.values.astype(float)
    user_ids = ui.index.values
    item_ids = ui.columns.values

    id_to_name = anime.set_index('anime_id')['name'].to_dict()

    return M, user_ids, item_ids, id_to_name


def cosine_item_item(M: np.ndarray):
    norms = np.linalg.norm(M, axis=0)
    norms[norms == 0] = 1e-9 
    
    S = (M.T @ M) / np.outer(norms, norms)
    np.fill_diagonal(S, 0.0)  # no queremos recomendarnos a nosotros mismos
    return S


def find_anime_id_by_name(anime_df: pd.DataFrame, name: str):
    n = name.strip().lower()
    exact = anime_df[anime_df['name'].str.lower() == n]
    if not exact.empty:
        return int(exact.iloc[0]['anime_id'])
    contains = anime_df[anime_df['name'].str.lower().str.contains(n, na=False)]
    if not contains.empty:
        return int(contains.iloc[0]['anime_id'])
    return None


def top_similar_to_title(title: str,
                         anime_df: pd.DataFrame,
                         item_ids: np.ndarray,
                         S: np.ndarray,
                         id_to_name: dict,
                         k: int = 10) -> pd.DataFrame:
    aid = find_anime_id_by_name(anime_df, title)
    if aid is None:
        raise ValueError(f"No encontré '{title}' en anime.csv.")
    if aid not in item_ids:
        raise ValueError(f"'{title}' existe pero fue filtrado por MIN_RATINGS_PER_ITEM. "
                         f"Baja ese valor o elige otro título.")
    j = int(np.where(item_ids == aid)[0][0])
    sims = S[j]
    top_idx = np.argsort(-sims)[:k]
    rows = []
    for idx in top_idx:
        aid2 = int(item_ids[idx])
        rows.append({
            "anime_id": aid2,
            "name": id_to_name.get(aid2, str(aid2)),
            "similarity": float(sims[idx])
        })
    return pd.DataFrame(rows)


def recommend_for_user(user_id: int,
                       M: np.ndarray,
                       user_ids: np.ndarray,
                       item_ids: np.ndarray,
                       S: np.ndarray,
                       id_to_name: dict,
                       k: int = 10) -> pd.DataFrame:
    if user_id not in user_ids:
        raise ValueError(f"Usuario {user_id} no encontrado en rating.csv (o quedó filtrado).")
    i = int(np.where(user_ids == user_id)[0][0])
    r = M[i] 

    scores = S @ r

    seen = r > 0
    scores[seen] = -np.inf

    top_idx = np.argsort(-scores)[:k]
    rows = []
    for idx in top_idx:
        aid = int(item_ids[idx])
        rows.append({
            "anime_id": aid,
            "name": id_to_name.get(aid, str(aid)),
            "score": float(scores[idx])
        })
    return pd.DataFrame(rows)


def main():
    anime, ratings = load_data(ANIME_CSV, RATINGS_CSV)
    M, user_ids, item_ids, id_to_name = build_user_item_matrix(
        anime, ratings, min_ratings_per_item=MIN_RATINGS_PER_ITEM
    )
    S = cosine_item_item(M)

    if ACTION == "similar_to":
        out = top_similar_to_title(SIMILAR_TO_TITLE, anime, item_ids, S, id_to_name, k=TOP_K)
        print("\n=== ÍTEMS SIMILARES ===")
        print(out.to_string(index=False))
    elif ACTION == "recommend_for_user":
        out = recommend_for_user(TARGET_USER_ID, M, user_ids, item_ids, S, id_to_name, k=TOP_K)
        print("\n=== RECOMENDACIONES PARA USUARIO ===")
        print(out.to_string(index=False))
    else:
        print("ACTION debe ser 'similar_to' o 'recommend_for_user'.")


if __name__ == "__main__":
    main()
