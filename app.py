# app.py
import pickle
from pathlib import Path

import streamlit as st
import pandas as pd

from src.content_based import ContentBasedRecommender
from src.hybrid import HybridRecommender

st.set_page_config(page_title="RecoSys from Scratch", layout="wide")
st.title("🎥 Recommendation System from Scratch")
st.markdown("**Matrix Factorization (SGD) + Content-Based Hybrid** | Portfolio Project")

# Load data for display (cheap, changes rarely -> cache_data is fine)
@st.cache_data
def load_data():
    ratings = pd.read_parquet("data/ratings.parquet")
    movies = pd.read_parquet("data/movies.parquet")
    return ratings, movies

# Load models once per session -> cache_resource (not cache_data, since these
# are stateful model objects, not serializable data)
@st.cache_resource
def load_models(ratings: pd.DataFrame, movies: pd.DataFrame):
    mf_path = Path("src/mf_model.pkl")
    if not mf_path.exists():
        raise FileNotFoundError(
            "src/mf_model.pkl not found. Run `python train.py` locally and "
            "commit the file, or see README Training section."
        )
    with open(mf_path, "rb") as f:
        mf_model = pickle.load(f)

    content_model = ContentBasedRecommender()
    content_model.fit(movies)

    hybrid_recommender = HybridRecommender(mf_model, content_model, ratings)
    return hybrid_recommender

ratings, movies = load_data()
hybrid_recommender = load_models(ratings, movies)

# Sidebar
st.sidebar.header("Demo Controls")
user_id = st.sidebar.number_input("User ID", min_value=1, max_value=943, value=196)
top_k = st.sidebar.slider("Top-K Recommendations", 5, 20, 10)
cold_start_mode = st.sidebar.checkbox("Cold-Start Mode (New User)", value=False)
show_explanations = st.sidebar.checkbox("Show explanations", value=False,
                                         help="Why was each movie recommended: "
                                              "collab/content scores, blend weight, "
                                              "dominant latent factor, matching genres.")

if st.sidebar.button("Get Recommendations"):
    with st.spinner("Fetching recommendations..."):
        try:
            liked_movie_ids = [1, 2, 50] if cold_start_mode else None

            recs = hybrid_recommender.recommend(
                user_id=user_id,
                top_k=top_k,
                liked_movies_for_coldstart=liked_movie_ids,
                explain=show_explanations and not cold_start_mode,
                # Cold-start recs come from ContentBasedRecommender directly,
                # which doesn't build the same explanation dict -- keep the
                # toggle honest rather than silently ignoring it.
            )

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Your Input History (Sample)")
                user_history = ratings[ratings['user_id'] == user_id].head(5)
                if not user_history.empty:
                    hist = user_history.merge(movies, on='movie_id')
                    st.dataframe(hist[['title', 'rating']], hide_index=True)
                else:
                    st.info("New user - Cold Start Active")

            with col2:
                st.subheader("🟢 Top Recommendations")
                if recs:
                    if show_explanations and not cold_start_mode:
                        for r in recs:
                            exp = r.get('explanation', {})
                            with st.expander(f"{r['title']}"):
                                c1, c2, c3 = st.columns(3)
                                c1.metric("Collab score", exp.get('collaborative_score'))
                                c2.metric("Content score", exp.get('content_score'))
                                c3.metric("Collab weight", exp.get('collaborative_weight'))
                                factor = exp.get('dominant_latent_factor')
                                if factor:
                                    st.caption(f"Dominant latent factor: #{factor['index']} "
                                               f"(contribution {factor['contribution']})")
                                genres = exp.get('matching_genres')
                                if genres:
                                    st.caption(f"Shared genres with your taste profile: {', '.join(genres)}")
                    else:
                        rec_df = pd.DataFrame(recs)[['movie_id', 'title']]
                        st.dataframe(rec_df, hide_index=True)
                        if cold_start_mode and show_explanations:
                            st.caption("Explanations aren't available for cold-start "
                                       "recommendations (no MF signal exists yet for a new user).")
                else:
                    st.warning("No recommendations returned")

        except Exception as e:
            st.error(f"Recommendation error: {e}")

st.markdown("---")
st.subheader("🎯 Find Similar Movies")
st.caption("Item-to-item content similarity (genres + year) -- independent of any user, "
           "like a 'more like this' widget on a movie's own page.")

movie_options = movies[['movie_id', 'title']].sort_values('title')
selected_title = st.selectbox("Pick a movie", movie_options['title'])
if st.button("Find Similar"):
    selected_id = int(movie_options.loc[movie_options['title'] == selected_title, 'movie_id'].iloc[0])
    similar = hybrid_recommender.content_model.get_similar_movies(selected_id, top_k=8)
    if similar:
        st.dataframe(pd.DataFrame(similar), hide_index=True)
    else:
        st.warning("No similar movies found")

st.markdown("---")
st.info("**Key Features Demonstrated**: SGD Matrix Factorization from scratch, "
        "Confidence-weighted Hybrid, Cold-start fallback, Time-based eval, "
        "Explainable recommendations, Content-based item similarity")