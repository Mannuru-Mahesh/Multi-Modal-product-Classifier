"""
feature_extraction.py
------------------------
Extracts numeric feature vectors from each modality:

  IMAGE features (per image):
    - Color histogram (8 bins per RGB channel = 24 features)
      -> captures the overall color palette
    - HOG (Histogram of Oriented Gradients)
      -> captures shape/edge information independent of color

  TEXT features (per description):
    - TF-IDF vectors over the description corpus
      -> captures which words/phrases are used
    - Reduced to a fixed-size dense vector with TruncatedSVD
      (similar in spirit to a text embedding)

These hand-engineered features are a practical, GPU-free stand-in
for what a CNN (image) and a transformer (text/CLIP) would learn
automatically. The MULTI-MODAL FUSION concept — combining two
different feature spaces into one classifier — is identical
regardless of how each feature vector was produced.
"""

import numpy as np
from PIL import Image
from skimage.feature import hog
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


# =================================================================
# IMAGE FEATURES
# =================================================================

def color_histogram(img_array, bins=8):
    """
    Compute a normalized color histogram for an RGB image.
    Returns a flat vector of length bins*3 (8*3 = 24).
    """
    hist = []
    for channel in range(3):  # R, G, B
        h, _ = np.histogram(img_array[:, :, channel], bins=bins, range=(0, 255))
        h = h / h.sum()  # normalize so it's scale-invariant
        hist.extend(h)
    return np.array(hist)


def hog_features(img_array):
    """
    Compute HOG features on the grayscale version of the image.
    HOG captures EDGES and SHAPES — e.g., a table's straight lines
    vs. a star's pointed edges — regardless of color.
    """
    gray = np.dot(img_array[:, :, :3], [0.299, 0.587, 0.114])  # RGB -> grayscale
    features = hog(
        gray,
        orientations=8,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        feature_vector=True
    )
    return features


def extract_image_features(image_paths, base_dir='data'):
    """
    Loop over a list of image paths and extract [color_hist + HOG]
    for each one. Returns a 2D numpy array (n_samples, n_features).
    """
    feats = []
    for rel_path in image_paths:
        img = Image.open(f"{base_dir}/{rel_path}").convert('RGB')
        arr = np.array(img)
        color_feat = color_histogram(arr)
        hog_feat = hog_features(arr)
        feats.append(np.concatenate([color_feat, hog_feat]))
    return np.array(feats)


# =================================================================
# TEXT FEATURES
# =================================================================

def fit_text_pipeline(descriptions, n_components=40):
    """
    Fit a TF-IDF vectorizer + TruncatedSVD on the training
    descriptions. SVD reduces TF-IDF's high-dimensional sparse
    vectors to a small dense "embedding" — easier to combine
    with image features and to feed into a NumPy MLP.

    Returns the fitted (vectorizer, svd) so the SAME transforms
    can be applied to validation/test/new descriptions later
    (critical to avoid data leakage).
    """
    vectorizer = TfidfVectorizer(stop_words='english', max_features=300)
    tfidf = vectorizer.fit_transform(descriptions)

    # n_components can't exceed n_features or n_samples
    n_components = min(n_components, tfidf.shape[1] - 1, tfidf.shape[0] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    svd.fit(tfidf)

    return vectorizer, svd


def extract_text_features(descriptions, vectorizer, svd):
    """Apply a FITTED vectorizer + SVD to new descriptions."""
    tfidf = vectorizer.transform(descriptions)
    return svd.transform(tfidf)
