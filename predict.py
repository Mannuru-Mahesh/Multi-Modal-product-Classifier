"""
predict.py
-----------
Loads all trained models (image-only, text-only, early-fusion,
multi-modal MLP) and classifies new products from:
  - An image file path (or a random test image from the dataset)
  - A text description string

Usage:
    python src/predict.py

You can also import predict_product() into another script.
"""

import numpy as np
import pandas as pd
import joblib
from PIL import Image
import sys
import os

sys.path.insert(0, 'src')
from feature_extraction import extract_image_features, extract_text_features
from numpy_mlp import MultiModalMLP


# =================================================================
# LOAD ALL MODELS + PREPROCESSORS
# =================================================================
def load_artifacts():
    """
    Load saved models and preprocessors from the models/ directory.
    Returns a dict with all artifacts.
    """
    print("Loading models...")
    preprocessors = joblib.load('models/preprocessors.pkl')
    image_model   = joblib.load('models/image_model.pkl')
    text_model    = joblib.load('models/text_model.pkl')
    fusion_model  = joblib.load('models/fusion_rf_model.pkl')
    mlp           = MultiModalMLP.load('models/multimodal_mlp.npz')

    return {
        'preprocessors': preprocessors,
        'image_model':   image_model,
        'text_model':    text_model,
        'fusion_model':  fusion_model,
        'mlp':           mlp,
    }


# =================================================================
# SINGLE PREDICTION FUNCTION
# =================================================================
def predict_product(image_path, description, artifacts):
    """
    Predict the product category for one item using all 4 models.

    Parameters
    ----------
    image_path  : str  — path to a product image (PNG/JPEG)
    description : str  — free-text product description
    artifacts   : dict — loaded models and preprocessors

    Returns
    -------
    dict with predictions from each model + confidence scores
    """
    pp = artifacts['preprocessors']
    le = pp['label_encoder']
    img_scaler = pp['img_scaler']
    txt_scaler = pp['txt_scaler']
    vectorizer = pp['vectorizer']
    svd        = pp['svd']

    # --- Extract features ---
    # Image features: color histogram + HOG
    # Pass the full path directly; we override base_dir to empty
    # and pass the full path string as the "relative" path
    img = Image.open(image_path).convert('RGB')
    arr = np.array(img)
    from feature_extraction import color_histogram, hog_features
    img_vec = np.concatenate([color_histogram(arr), hog_features(arr)]).reshape(1, -1)
    img_feats = img_scaler.transform(img_vec)

    # Text features: TF-IDF + SVD (using FITTED transformers, not re-fitting)
    txt_feats = extract_text_features([description], vectorizer, svd)
    txt_feats = txt_scaler.transform(txt_feats)

    # Fused features (early fusion)
    fused_feats = np.concatenate([img_feats, txt_feats], axis=1)

    results = {}

    # --- Image-only prediction ---
    img_pred = artifacts['image_model'].predict(img_feats)[0]
    img_prob = artifacts['image_model'].predict_proba(img_feats)[0]
    results['Image Only'] = {
        'prediction': le.inverse_transform([img_pred])[0],
        'confidence': f"{img_prob.max()*100:.1f}%",
        'all_probs': dict(zip(le.classes_, img_prob.round(3)))
    }

    # --- Text-only prediction ---
    txt_pred = artifacts['text_model'].predict(txt_feats)[0]
    txt_prob = artifacts['text_model'].predict_proba(txt_feats)[0]
    results['Text Only'] = {
        'prediction': le.inverse_transform([txt_pred])[0],
        'confidence': f"{txt_prob.max()*100:.1f}%",
        'all_probs': dict(zip(le.classes_, txt_prob.round(3)))
    }

    # --- Early fusion prediction ---
    fuse_pred = artifacts['fusion_model'].predict(fused_feats)[0]
    fuse_prob = artifacts['fusion_model'].predict_proba(fused_feats)[0]
    results['Early Fusion'] = {
        'prediction': le.inverse_transform([fuse_pred])[0],
        'confidence': f"{fuse_prob.max()*100:.1f}%",
        'all_probs': dict(zip(le.classes_, fuse_prob.round(3)))
    }

    # --- Multi-modal MLP prediction ---
    mlp_probs  = artifacts['mlp'].forward(img_feats, txt_feats)[0]
    mlp_pred   = np.argmax(mlp_probs)
    results['Multi-Modal MLP'] = {
        'prediction': le.inverse_transform([mlp_pred])[0],
        'confidence': f"{mlp_probs.max()*100:.1f}%",
        'all_probs': dict(zip(le.classes_, mlp_probs.round(3)))
    }

    return results


# =================================================================
# DEMO: run predictions on 3 hand-picked test cases
# =================================================================
if __name__ == '__main__':
    artifacts = load_artifacts()
    df = pd.read_csv('data/metadata.csv')
    le = artifacts['preprocessors']['label_encoder']

    print("\n" + "=" * 65)
    print("MULTI-MODAL PRODUCT CLASSIFIER — Live Predictions")
    print("=" * 65)

    # Pick 3 interesting test cases:
    # 1. Clean sample (should be easy for all models)
    # 2. Occluded image only (image-only model should struggle)
    # 3. Generic text only (text-only model should struggle)
    test_cases = []

    # Case 1: clean image AND clean text
    clean = df[(~df['image_occluded']) & (~df['text_generic'])].iloc[0]
    test_cases.append(('Clean — both modalities informative', clean))

    # Case 2: occluded image + clean text
    occ_img = df[df['image_occluded'] & (~df['text_generic'])].iloc[0]
    test_cases.append(('Hard — image occluded (random static), text OK', occ_img))

    # Case 3: clean image + generic text
    gen_txt = df[(~df['image_occluded']) & df['text_generic']].iloc[0]
    test_cases.append(('Hard — text is generic boilerplate, image OK', gen_txt))

    for case_name, row in test_cases:
        print(f"\n{'─'*65}")
        print(f"Case: {case_name}")
        print(f"True category : {row['category']}")
        print(f"Description   : \"{row['description']}\"")
        print(f"Image         : {row['image_path']}")
        print()

        results = predict_product(
            image_path=f"data/{row['image_path']}",
            description=row['description'],
            artifacts=artifacts
        )

        for model_name, res in results.items():
            correct = "✅" if res['prediction'] == row['category'] else "❌"
            print(f"  {correct} {model_name:20s} → {res['prediction']:12s}  (confidence: {res['confidence']})")

    print(f"\n{'─'*65}")
    print("\nKey takeaway:")
    print("  When one modality is uninformative (occluded image or generic")
    print("  text), single-modality models guess randomly but fusion models")
    print("  lean on whichever modality IS informative and stay accurate.")
