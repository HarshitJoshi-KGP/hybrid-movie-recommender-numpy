# src/evaluate.py
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def time_based_split(ratings_df: pd.DataFrame, test_ratio=0.2):
    """More realistic split: train on older ratings, test on newer"""
    ratings_sorted = ratings_df.sort_values('timestamp')
    split_idx = int(len(ratings_sorted) * (1 - test_ratio))

    train = ratings_sorted.iloc[:split_idx].copy()
    test = ratings_sorted.iloc[split_idx:].copy()

    print(f"Time-based split: Train {len(train):,} ratings | Test {len(test):,} ratings")
    print(f"Train period: {pd.to_datetime(train['timestamp'], unit='s').min()} to {pd.to_datetime(train['timestamp'], unit='s').max()}")
    return train, test


def compute_rating_metrics(y_true, y_pred):
    """RMSE, MAE for rating prediction"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return {'RMSE': rmse, 'MAE': mae}


def precision_at_k(recommended, relevant, k=10):
    """Precision@K. Denominator is always k (standard definition) rather than
    min(k, len(rec_set)), which understated the penalty when fewer than k
    unique items were returned."""
    rec_set = set(recommended[:k])
    rel_set = set(relevant)
    return len(rec_set & rel_set) / k


def ndcg_at_k(recommended, relevant, k=10):
    """NDCG@K (simplified)"""
    dcg = 0.0
    idcg = 0.0
    for i, item in enumerate(recommended[:k]):
        if item in relevant:
            dcg += 1.0 / np.log2(i + 2)
    for i in range(min(k, len(relevant))):
        idcg += 1.0 / np.log2(i + 2)
    return dcg / idcg if idcg > 0 else 0.0


class Evaluator:
    def __init__(self, mf_model, hybrid_recommender, ratings_df):
        self.mf_model = mf_model
        self.hybrid = hybrid_recommender
        self.ratings_df = ratings_df

    def evaluate_baselines(self, test_df: pd.DataFrame):
        results = {}

        # Baseline 1: Global Average
        global_avg = self.ratings_df['rating'].mean()
        y_pred_global = [global_avg] * len(test_df)
        results['Global Average'] = compute_rating_metrics(test_df['rating'], y_pred_global)

        # Baseline 2: Item Popularity (mean rating per item)
        item_means = self.ratings_df.groupby('movie_id')['rating'].mean()
        y_pred_item = test_df['movie_id'].map(item_means).fillna(global_avg)
        results['Item Popularity'] = compute_rating_metrics(test_df['rating'], y_pred_item)

        # Baseline 3: Pure MF (collaborative)
        y_pred_mf = []
        for _, row in test_df.iterrows():
            pred = self.mf_model.predict(row['user_id'], row['movie_id'])
            y_pred_mf.append(pred)
        results['Matrix Factorization (MF)'] = compute_rating_metrics(test_df['rating'], y_pred_mf)

        # Baseline 4: Hybrid (MF + content, confidence-weighted)
        # Previously this row just re-ran the MF prediction loop ("for eval we
        # can use MF + content boost later") so the reported Hybrid numbers
        # were actually pure-MF numbers. It now calls the hybrid model's own
        # predict_rating(), which blends in the content-based score.
        y_pred_hybrid = []
        for _, row in test_df.iterrows():
            try:
                pred = self.hybrid.predict_rating(row['user_id'], row['movie_id'])
            except Exception:
                pred = global_avg
            y_pred_hybrid.append(pred)
        results['Hybrid (MF + Content)'] = compute_rating_metrics(test_df['rating'], y_pred_hybrid)

        # Convert to DataFrame for nice display
        df_results = pd.DataFrame(results).T
        print("\n=== Rating Prediction Metrics ===")
        print(df_results.round(4))

        return df_results

    def evaluate_ranking(self, test_df: pd.DataFrame, k=10):
        """Ranking metrics - more important for real recommenders"""
        print(f"\nComputing ranking metrics @K={k} (this may take a moment)...")

        user_groups = test_df.groupby('user_id')
        precisions = []
        ndcgs = []

        for user_id, group in user_groups:
            if len(group) < 5:  # skip users with too few test ratings
                continue

            # Get ground truth relevant items
            relevant = group['movie_id'].tolist()

            # Get recommendations from hybrid
            recs = self.hybrid.recommend(user_id=user_id, top_k=k * 2)
            recommended = [r['movie_id'] for r in recs]

            precisions.append(precision_at_k(recommended, relevant, k))
            ndcgs.append(ndcg_at_k(recommended, relevant, k))

        print(f"Avg Precision@{k}: {np.mean(precisions):.4f}")
        print(f"Avg NDCG@{k}: {np.mean(ndcgs):.4f}")

        return {'Precision@K': np.mean(precisions), 'NDCG@K': np.mean(ndcgs)}


# Quick run
if __name__ == "__main__":
    from data.preprocess import load_data
    from src.model import MatrixFactorizationSGD
    from src.content_based import ContentBasedRecommender
    from src.hybrid import HybridRecommender

    ratings, movies = load_data()
    train, test = time_based_split(ratings)

    mf_model = MatrixFactorizationSGD.load("src/mf_model.pkl")
    content_model = ContentBasedRecommender()
    content_model.fit(movies)
    hybrid = HybridRecommender(mf_model, content_model, train)

    evaluator = Evaluator(mf_model, hybrid, ratings)
    evaluator.evaluate_baselines(test)
    evaluator.evaluate_ranking(test, k=10)
