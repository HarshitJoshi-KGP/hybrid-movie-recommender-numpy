# train.py
"""
Dedicated training entry point.

Why this file exists: running `python -m src.model` trains the model fine,
but Python's pickle records a class as belonging to whichever module was
__main__ at save time. That made the saved model only loadable from that
same entry point -- loading it from src/api.py (a different __main__)
failed with:

    AttributeError: Can't get attribute 'MatrixFactorizationSGD' on
    <module '__main__' ...>

Importing MatrixFactorizationSGD normally (as src.model.MatrixFactorizationSGD)
here, instead of defining/running it as __main__, makes the pickle portable
across every other entry point (api.py, evaluate.py, test.py, notebooks, etc).
"""
from pathlib import Path
import matplotlib.pyplot as plt

from data.preprocess import load_data
from src.model import MatrixFactorizationSGD

if __name__ == "__main__":
    ratings, _ = load_data()

    model = MatrixFactorizationSGD(n_factors=20, lr=0.01, reg=0.02, n_epochs=15)
    model.fit(ratings)

    Path("notebooks").mkdir(exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(model.train_loss_history)
    plt.title("Training Loss (Regularized MSE)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("notebooks/loss_curve.png")
    print("Loss curve saved to notebooks/loss_curve.png")

    model.save("src/mf_model.pkl")
    print("Model saved to src/mf_model.pkl")
