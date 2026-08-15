# data/preprocess.py
import zipfile
import urllib.request
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

def download_movielens_100k():
    url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
    zip_path = DATA_DIR / "ml-100k.zip"
    extract_dir = DATA_DIR / "ml-100k"
    
    if not extract_dir.exists():
        print("Downloading MovieLens 100K...")
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
        print("Download complete.")
    return extract_dir

GENRE_NAMES = [
    'unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy',
    'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
    'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
]

def load_data(force_download: bool = False):
    """Load ratings + movies as DataFrames.

    If data/ratings.parquet and data/movies.parquet already exist (i.e.
    someone already ran this module's __main__ block once), reuse them
    instead of re-downloading and re-parsing the raw MovieLens files on
    every call -- callers like scripts/tune.py and tests/test_smoke.py call
    load_data() repeatedly, and the parquet files are the canonical
    already-processed output of this exact function. Pass
    force_download=True to bypass the cache and re-derive from source.
    """
    ratings_path = DATA_DIR / "ratings.parquet"
    movies_path = DATA_DIR / "movies.parquet"
    if not force_download and ratings_path.exists() and movies_path.exists():
        ratings = pd.read_parquet(ratings_path)
        movies = pd.read_parquet(movies_path)
        print(f"Loaded {len(ratings):,} ratings and {len(movies):,} movies from cached parquet files")
        return ratings, movies

    extract_dir = download_movielens_100k()
    
    # Load ratings
    ratings = pd.read_csv(
        extract_dir / "u.data",
        sep='\t',
        names=['user_id', 'movie_id', 'rating', 'timestamp'],
        engine='python'
    )
    
    # Load movies (for later content-based)
    movies = pd.read_csv(
        extract_dir / "u.item",
        sep='|',
        names=['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url'] + GENRE_NAMES,
        encoding='latin-1',
        engine='python'
    )

    # Keep title + the 19 real genre flags from u.item (previously these were
    # dropped and content-based similarity ran on titles only)
    movies = movies[['movie_id', 'title'] + GENRE_NAMES]

    # Extract release year as a light extra content signal (e.g. "Toy Story (1995)")
    movies['year'] = movies['title'].str.extract(r'\((\d{4})\)').astype('float')
    
    print(f"Loaded {len(ratings):,} ratings from {ratings['user_id'].nunique():,} users and {ratings['movie_id'].nunique():,} movies")
    print(f"Rating range: {ratings['rating'].min()} - {ratings['rating'].max()}")
    
    return ratings, movies

if __name__ == "__main__":
    # Running this file directly means "regenerate the parquet files from
    # source", so bypass the cache load_data() would otherwise take.
    ratings, movies = load_data(force_download=True)
    ratings.to_parquet(DATA_DIR / "ratings.parquet", index=False)
    movies.to_parquet(DATA_DIR / "movies.parquet", index=False)
    print("Data saved to parquet files.")