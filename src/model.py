# src/model.py
import numpy as np
import pandas as pd
import pickle


class MatrixFactorizationSGD:
    """
    Matrix Factorization with SGD + biases + L2 regularization.
    Implemented from scratch (no surprise/implicit).
    """

    def __init__(self, n_factors=20, lr=0.01, reg=0.02, n_epochs=20, random_state=42):
        self.n_factors = n_factors      # latent factors (k)
        self.lr = lr                    # learning rate
        self.reg = reg                  # L2 regularization
        self.n_epochs = n_epochs
        self.random_state = random_state

        self.global_bias = None
        self.user_bias = None
        self.item_bias = None
        self.user_factors = None
        self.item_factors = None

        self.train_loss_history = []

    def fit(self, ratings_df: pd.DataFrame):
        """Train using SGD on (user_id, movie_id, rating)"""
        np.random.seed(self.random_state)

        # Map to 0-based indices
        user_ids = ratings_df['user_id'].unique()
        item_ids = ratings_df['movie_id'].unique()

        self.user_map = {uid: i for i, uid in enumerate(user_ids)}
        self.item_map = {iid: i for i, iid in enumerate(item_ids)}

        n_users = len(user_ids)
        n_items = len(item_ids)

        # Initialize parameters
        self.global_bias = ratings_df['rating'].mean()
        self.user_bias = np.zeros(n_users)
        self.item_bias = np.zeros(n_items)
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))

        # Convert to raw numpy arrays *once*, up front. SGD updates are
        # inherently sequential (each example's update depends on the
        # previous one), so this can't become a single batched matrix op
        # without changing the algorithm -- but the original loop paid for
        # a pandas Series allocation on every single row via `.iterrows()`,
        # which dominates the runtime far more than the actual arithmetic
        # does. Pulling out plain int/float numpy arrays and indexing them
        # directly in the loop removes that overhead entirely.
        user_idx_all = ratings_df['user_id'].map(self.user_map).to_numpy(dtype=np.int64)
        item_idx_all = ratings_df['movie_id'].map(self.item_map).to_numpy(dtype=np.int64)
        rating_all = ratings_df['rating'].to_numpy(dtype=np.float64)
        n_ratings = len(rating_all)

        print(f"Training on {n_ratings} ratings | Users: {n_users} | Items: {n_items}")

        rng = np.random.default_rng(self.random_state)

        for epoch in range(self.n_epochs):
            # Shuffle indices instead of the DataFrame
            perm = rng.permutation(n_ratings)

            epoch_loss = 0.0

            for idx in perm:
                u = user_idx_all[idx]
                i = item_idx_all[idx]
                r_true = rating_all[idx]

                # Prediction
                pred = (self.global_bias +
                        self.user_bias[u] +
                        self.item_bias[i] +
                        np.dot(self.user_factors[u], self.item_factors[i]))

                error = r_true - pred

                # Update biases
                self.user_bias[u] += self.lr * (error - self.reg * self.user_bias[u])
                self.item_bias[i] += self.lr * (error - self.reg * self.item_bias[i])

                # Update latent factors
                user_grad = error * self.item_factors[i] - self.reg * self.user_factors[u]
                item_grad = error * self.user_factors[u] - self.reg * self.item_factors[i]

                self.user_factors[u] += self.lr * user_grad
                self.item_factors[i] += self.lr * item_grad

                epoch_loss += error ** 2

            avg_loss = epoch_loss / n_ratings
            self.train_loss_history.append(avg_loss)

            if epoch % 5 == 0 or epoch == self.n_epochs - 1:
                print(f"Epoch {epoch+1}/{self.n_epochs} - Loss: {avg_loss:.4f}")

        print("Training completed.")
        return self

    def predict(self, user_id, movie_id):
        """Predict rating for a user-movie pair"""
        if user_id not in self.user_map or movie_id not in self.item_map:
            return self.global_bias  # cold-start fallback

        u = self.user_map[user_id]
        i = self.item_map[movie_id]

        pred = (self.global_bias +
                self.user_bias[u] +
                self.item_bias[i] +
                np.dot(self.user_factors[u], self.item_factors[i]))
        return np.clip(pred, 1.0, 5.0)  # ratings are 1-5

    def get_user_embedding(self, user_id):
        """Return user latent factors (for hybrid / similarity use cases)."""
        if user_id not in self.user_map:
            return None
        return self.user_factors[self.user_map[user_id]]

    def get_item_embedding(self, movie_id):
        """Return item latent factors (for hybrid / similarity use cases)."""
        if movie_id not in self.item_map:
            return None
        return self.item_factors[self.item_map[movie_id]]

    def save(self, path: str = "model.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str = "model.pkl"):
        with open(path, "rb") as f:
            return pickle.load(f)


# NOTE: training is done via train.py at the project root, not from here.
# A pickle records a class under whichever module was __main__ at save time,
# so training by running this file directly (`python -m src.model` /
# `python src/model.py`) would save the model as `__main__.MatrixFactorizationSGD`
# instead of `src.model.MatrixFactorizationSGD`, making it fail to load from
# any other entry point (this bit the API server -- see train.py's docstring).
# Run `python train.py` to train.
