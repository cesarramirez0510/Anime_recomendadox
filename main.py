import argparse
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer



def load_data(anime_path: str, ratings_path: str):
    anime_cols = ['anime_id','name','genre','type','episodes','rating','members']
    rating_cols = ['user_id','anime_id','rating']

    anime = pd.read_csv(anime_path, names=anime_cols, header=0)
    ratings = pd.read_csv(ratings_path, names=rating_cols, header=0)

    ratings = ratings[ratings['rating'] != -1].copy()

    anime['name_norm'] = anime['name'].str.strip().str.lower()

    return anime, ratings

def build_item_user_matrix(anime, ratings, min_ratings_per_item=5):
    item_counts = ratings['anime_id'].value_counts()
    keep_items = item_counts[item_counts >= min_ratings_per_item].index
    ratings_f = ratings[ratings['anime_id'].isin(keep_items)].copy()

    unique_items = np.sort(ratings_f['anime_id'].unique())
    unique_users = np.sort(ratings_f['user_id'].unique())

    item_to_idx = {aid:i for i, aid in enumerate(unique_items)}
    user_to_idx = {uid:i for i, uid in enumerate(unique_users)}

    rows = ratings_f['anime_id'].map(item_to_idx).values
    cols = ratings_f['user_id'].map(user_to_idx).values
    vals = ratings_f['rating'].astype(float).values

    user_means = pd.Series(vals, index=cols).groupby(level=0).mean()
    centered_vals = vals - user_means[cols].values

    mat = csr_matrix((centered_vals, (rows, cols)),
                     shape=(len(unique_items), len(unique_users)))

    meta = anime[anime['anime_id'].isin(unique_items)].copy()
    meta = meta.set_index('anime_id').loc[unique_items].reset_index()

    return mat, meta, item_to_idx

def build_content_similarity(meta):
    genres = meta['genre'].fillna('').astype(str).values
    tfidf = TfidfVectorizer(token_pattern=r'[^,\s]+' , lowercase=True)
    G = tfidf.fit_transform(genres)
    sim_content = cosine_similarity(G, dense_output=False)  # sparse
    return sim_content

def build_collab_similarity(item_user_mat):
    sim = cosine_similarity(item_user_mat, dense_output=False)
    return sim

def fuse_similarities(sim_collab, sim_content=None, alpha=1.0):
    if sim_content is None:
        return sim_collab
    return (sim_collab.multiply(alpha)) + (sim_content.multiply(1 - alpha))

def top_similar_items(anime_name, meta, sim_matrix, top_k=10):
    name_norm = anime_name.strip().lower()
    idx_series = meta['name_norm'] if 'name_norm' in meta else meta['name'].str.lower()
    matches = np.where(idx_series.values == name_norm)[0]
    if len(matches) == 0:
        contains = np.where(idx_series.str.contains(name_norm, regex=False).values)[0]
        if len(contains) == 0:
            raise ValueError(f"No encontré el animé: {anime_name}")
        target_idx = contains[0]
    else:
        target_idx = matches[0]

    sims = sim_matrix.getrow(target_idx).toarray().ravel()
    sims[target_idx] = -np.inf  # excluir el mismo item
    top_idx = np.argpartition(-sims, range(top_k))[:top_k]
    top_idx = top_idx[np.argsort(-sims[top_idx])]

    return meta.iloc[top_idx][['anime_id','name','genre']].assign(similarity=sims[top_idx])

def recommend_for_user(user_id, ratings, meta, sim_matrix, top_k=10, min_seen=1):
    seen = ratings[ratings['user_id'] == user_id]
    seen = seen[seen['rating'] != -1]
    if len(seen) < min_seen:
        raise ValueError(f"El usuario {user_id} tiene muy pocos ítems valorados.")

    aid_to_idx = {aid:i for i, aid in enumerate(meta['anime_id'].values)}
    seen = seen[seen['anime_id'].isin(aid_to_idx.keys())].copy()
    if seen.empty:
        raise ValueError("El usuario no coincide con los items de la matriz (tras filtrado).")

    scores = np.zeros(len(meta), dtype=float)
    seen_pairs = [(aid_to_idx[aid], r) for aid, r in zip(seen['anime_id'], seen['rating'])]

    for idx, r in seen_pairs:
        sims = sim_matrix.getrow(idx).toarray().ravel()
        scores += sims * float(r)

    seen_idx = [aid_to_idx[aid] for aid in seen['anime_id']]
    scores[seen_idx] = -np.inf

    top_idx = np.argpartition(-scores, range(top_k))[:top_k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    return meta.iloc[top_idx][['anime_id','name','genre']].assign(score=scores[top_idx])


def main():
    parser = argparse.ArgumentParser(description="Recomendador de animé basado en items (Kaggle).")
    parser.add_argument("--anime_csv", default="anime.csv")
    parser.add_argument("--ratings_csv", default="rating.csv")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Peso de similitud colaborativa (1.0=solo ratings, 0.0=solo géneros).")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--user_id", type=int, default=None,
                        help="Si se pasa, devuelve recomendaciones para este usuario.")
    parser.add_argument("--similar_to", type=str, default=None,
                        help="Si se pasa, devuelve ítems similares a este título (búsqueda exacta o contiene).")
    parser.add_argument("--min_ratings_per_item", type=int, default=5)
    args = parser.parse_args()

    anime, ratings = load_data(args.anime_csv, args.ratings_csv)

    item_user_mat, meta, item_to_idx = build_item_user_matrix(
        anime, ratings, min_ratings_per_item=args.min_ratings_per_item
    )

    sim_collab = build_collab_similarity(item_user_mat)
    sim_content = build_content_similarity(meta)

    sim_fused = fuse_similarities(sim_collab, sim_content, alpha=args.alpha)

    if args.similar_to:
        out = top_similar_items(args.similar_to, meta, sim_fused, top_k=args.topk)
        print("\n=== Ítems similares ===")
        print(out.to_string(index=False))
    elif args.user_id is not None:
        out = recommend_for_user(args.user_id, ratings, meta, sim_fused, top_k=args.topk)
        print("\n=== Recomendaciones para usuario ===")
        print(out.to_string(index=False))
    else:
        merged = pd.merge(anime[['anime_id','name']], ratings, on='anime_id')
        print(merged.head())

if __name__ == "__main__":
    main()
