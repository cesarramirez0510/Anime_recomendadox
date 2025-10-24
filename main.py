import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

ANIME_CSV = "anime.csv"
RATINGS_CSV = "rating.csv"

ACTION = "similar_to"      
SIMILAR_TO_TITLE = "Naruto"    
TARGET_USER_ID = 1             
TOP_K = 10                     
MIN_RATINGS_PER_ITEM = 50      

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
                           min_ratings_per_item: int = 50):
    
    print(f" Ratings iniciales: {len(ratings):,}")
    print(f" Usuarios únicos: {ratings['user_id'].nunique():,}")
    print(f" Animes únicos: {ratings['anime_id'].nunique():,}")
    
    if min_ratings_per_item > 1:
        counts = ratings['anime_id'].value_counts()
        keep = counts[counts >= min_ratings_per_item].index
        ratings = ratings[ratings['anime_id'].isin(keep)].copy()
        print(f"Después del filtro (≥{min_ratings_per_item} ratings): {len(ratings):,}")
    
    MIN_RATINGS_PER_USER = 5
    user_counts = ratings['user_id'].value_counts()
    active_users = user_counts[user_counts >= MIN_RATINGS_PER_USER].index
    ratings = ratings[ratings['user_id'].isin(active_users)].copy()
    print(f"Usuarios activos (≥{MIN_RATINGS_PER_USER} ratings): {ratings['user_id'].nunique():,}")

    user_ids = np.sort(ratings['user_id'].unique())
    item_ids = np.sort(ratings['anime_id'].unique())
    
    user_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
    item_to_idx = {iid: idx for idx, iid in enumerate(item_ids)}
    
    print(f" Dimensiones finales: {len(user_ids):,} usuarios  {len(item_ids):,} animes")
    
    row_indices = ratings['user_id'].map(user_to_idx).values
    col_indices = ratings['anime_id'].map(item_to_idx).values
    data = ratings['rating'].values
    
    M = csr_matrix(
        (data, (row_indices, col_indices)),
        shape=(len(user_ids), len(item_ids)),
        dtype=np.float32  
    )
    
    sparsity = 100 * (1 - M.nnz / (M.shape[0] * M.shape[1]))
    print(f" Sparsity: {sparsity:.2f}% (solo {M.nnz:,} valores no-cero)")
    
    id_to_name = anime.set_index('anime_id')['name'].to_dict()

    return M, user_ids, item_ids, id_to_name


def cosine_item_item(M: csr_matrix):
    """Calcula similitud coseno entre ítems usando matriz dispersa."""
    print("🔄 Calculando similitudes item-item...")
    
    S = cosine_similarity(M.T, dense_output=False)
    
    S = S.tocsr()
    S.setdiag(0)
    S.eliminate_zeros()
    
    print(f" Matriz de similitud: {S.shape[0]} × {S.shape[1]}")
    return S


def find_anime_id_by_name(anime_df: pd.DataFrame, name: str):
    """Busca un anime por nombre (exacto o parcial)."""
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
                         S: csr_matrix,
                         id_to_name: dict,
                         k: int = 10) -> pd.DataFrame:
    """Encuentra los k animes más similares a un título dado."""
    aid = find_anime_id_by_name(anime_df, title)
    if aid is None:
        raise ValueError(f" No encontré '{title}' en anime.csv.")
    if aid not in item_ids:
        raise ValueError(f" '{title}' existe pero fue filtrado por MIN_RATINGS_PER_ITEM. "
                         f"Baja ese valor o elige otro título.")
    
    j = int(np.where(item_ids == aid)[0][0])

    sims = S[j].toarray().flatten()
  
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
                       M: csr_matrix,
                       user_ids: np.ndarray,
                       item_ids: np.ndarray,
                       S: csr_matrix,
                       id_to_name: dict,
                       k: int = 10) -> pd.DataFrame:
    """Recomienda k ítems para un usuario basado en filtrado colaborativo."""
    if user_id not in user_ids:
        raise ValueError(f"Usuario {user_id} no encontrado en rating.csv (o quedó filtrado).")
    
    i = int(np.where(user_ids == user_id)[0][0])
    
    r = M[i].toarray().flatten()
    
    scores = S.dot(r)
    
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
    print("Iniciando sistema de recomendación de anime...\n")
    
    anime, ratings = load_data(ANIME_CSV, RATINGS_CSV)
    M, user_ids, item_ids, id_to_name = build_user_item_matrix(
        anime, ratings, min_ratings_per_item=MIN_RATINGS_PER_ITEM
    )
    S = cosine_item_item(M)

    print("\n" + "="*60)
    if ACTION == "similar_to":
        out = top_similar_to_title(SIMILAR_TO_TITLE, anime, item_ids, S, id_to_name, k=TOP_K)
        print(f"\n ANIMES SIMILARES A '{SIMILAR_TO_TITLE}'")
        print("="*60)
        print(out.to_string(index=False))
    elif ACTION == "recommend_for_user":
        out = recommend_for_user(TARGET_USER_ID, M, user_ids, item_ids, S, id_to_name, k=TOP_K)
        print(f"\nRECOMENDACIONES PARA USUARIO {TARGET_USER_ID}")
        print("="*60)
        print(out.to_string(index=False))
    else:
        print("ACTION debe ser 'similar_to' o 'recommend_for_user'.")


if __name__ == "__main__":
    main()