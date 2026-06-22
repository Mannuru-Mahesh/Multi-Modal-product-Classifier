"""
train.py
----------
Main pipeline for the Multi-Modal Product Classifier.

Steps:
  1. Load metadata (image paths, text descriptions, labels)
  2. Extract image features (color histogram + HOG) and text
     features (TF-IDF + SVD)
  3. Train/test split (stratified)
  4. Train THREE baseline models for comparison:
       a) Image-only  (Random Forest on image features)
       b) Text-only   (Logistic Regression on text features)
       c) Fusion (early)  (Random Forest on concatenated features)
  5. Train the custom NumPy multi-modal MLP (image branch + text
     branch + fusion head)
  6. Compare all models with accuracy + confusion matrices
  7. Save fitted feature extractors + trained models for inference

This script directly tests the core hypothesis of multi-modal
learning: FUSING modalities should outperform EITHER modality alone,
especially on the "noisy" examples where one modality is misleading.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from feature_extraction import extract_image_features, fit_text_pipeline, extract_text_features
from numpy_mlp import MultiModalMLP

sns.set_style('whitegrid')

# =================================================================
# 1. LOAD METADATA
# =================================================================
print("=" * 60)
print("STEP 1: Loading metadata")
print("=" * 60)

df = pd.read_csv('data/metadata.csv')
print(f"Total samples: {len(df)}")
print(df['category'].value_counts())

le = LabelEncoder()
y = le.fit_transform(df['category'])
print(f"\nClasses: {list(le.classes_)}")


# =================================================================
# 2. TRAIN/TEST SPLIT (stratified, on indices first — features
#    are extracted AFTER splitting where it matters, e.g. text
#    vectorizer fitting, to avoid leakage)
# =================================================================
print("\n" + "=" * 60)
print("STEP 2: Train/Test Split")
print("=" * 60)

idx_train, idx_test = train_test_split(
    np.arange(len(df)), test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(idx_train)} samples | Test: {len(idx_test)} samples")


# =================================================================
# 3. FEATURE EXTRACTION
# =================================================================
print("\n" + "=" * 60)
print("STEP 3: Feature Extraction")
print("=" * 60)

print("Extracting image features (color histogram + HOG)...")
X_img_all = extract_image_features(df['image_path'].values)
print(f"Image feature shape: {X_img_all.shape}")

print("Fitting text pipeline (TF-IDF + SVD) on TRAINING descriptions only...")
vectorizer, svd = fit_text_pipeline(df['description'].iloc[idx_train].values, n_components=40)
X_txt_all = extract_text_features(df['description'].values, vectorizer, svd)
print(f"Text feature shape: {X_txt_all.shape}")

# Scale image and text features separately (different ranges)
img_scaler = StandardScaler().fit(X_img_all[idx_train])
txt_scaler = StandardScaler().fit(X_txt_all[idx_train])

X_img_scaled = img_scaler.transform(X_img_all)
X_txt_scaled = txt_scaler.transform(X_txt_all)

# Split into train/test
X_img_train, X_img_test = X_img_scaled[idx_train], X_img_scaled[idx_test]
X_txt_train, X_txt_test = X_txt_scaled[idx_train], X_txt_scaled[idx_test]
y_train, y_test = y[idx_train], y[idx_test]

# Early-fusion feature set (simple concatenation)
X_fused_train = np.concatenate([X_img_train, X_txt_train], axis=1)
X_fused_test  = np.concatenate([X_img_test, X_txt_test], axis=1)


# =================================================================
# 4. BASELINE MODELS
# =================================================================
print("\n" + "=" * 60)
print("STEP 4: Training Baseline Models")
print("=" * 60)

results = {}
predictions = {}

# --- 4a. Image-only ---
img_model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
img_model.fit(X_img_train, y_train)
img_preds = img_model.predict(X_img_test)
results['Image Only (Random Forest)'] = accuracy_score(y_test, img_preds)
predictions['Image Only (Random Forest)'] = img_preds
print(f"Image Only (Random Forest):       accuracy = {results['Image Only (Random Forest)']:.3f}")

# --- 4b. Text-only ---
txt_model = LogisticRegression(max_iter=1000, random_state=42)
txt_model.fit(X_txt_train, y_train)
txt_preds = txt_model.predict(X_txt_test)
results['Text Only (Logistic Regression)'] = accuracy_score(y_test, txt_preds)
predictions['Text Only (Logistic Regression)'] = txt_preds
print(f"Text Only (Logistic Regression):  accuracy = {results['Text Only (Logistic Regression)']:.3f}")

# --- 4c. Early Fusion (concat features -> Random Forest) ---
fusion_rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
fusion_rf.fit(X_fused_train, y_train)
fusion_rf_preds = fusion_rf.predict(X_fused_test)
results['Early Fusion (Random Forest)'] = accuracy_score(y_test, fusion_rf_preds)
predictions['Early Fusion (Random Forest)'] = fusion_rf_preds
print(f"Early Fusion (Random Forest):     accuracy = {results['Early Fusion (Random Forest)']:.3f}")


# =================================================================
# 5. MULTI-MODAL NEURAL NETWORK (from scratch, NumPy)
# =================================================================
print("\n" + "=" * 60)
print("STEP 5: Training Multi-Modal Neural Network (NumPy)")
print("=" * 60)

mlp = MultiModalMLP(
    img_dim=X_img_train.shape[1],
    txt_dim=X_txt_train.shape[1],
    branch_hidden=32,
    fusion_hidden=32,
    n_classes=len(le.classes_),
    seed=42
)

history = mlp.fit(
    X_img_train, X_txt_train, y_train,
    epochs=200, batch_size=32, lr=0.05,
    X_img_val=X_img_test, X_txt_val=X_txt_test, y_val=y_test,
    verbose=True
)

mlp_preds = mlp.predict(X_img_test, X_txt_test)
results['Multi-Modal Neural Net (NumPy)'] = accuracy_score(y_test, mlp_preds)
predictions['Multi-Modal Neural Net (NumPy)'] = mlp_preds
print(f"\nMulti-Modal Neural Net (NumPy):    accuracy = {results['Multi-Modal Neural Net (NumPy)']:.3f}")


# =================================================================
# 6. COMPARISON & VISUALIZATION
# =================================================================
print("\n" + "=" * 60)
print("STEP 6: Model Comparison")
print("=" * 60)

results_df = pd.DataFrame(list(results.items()), columns=['Model', 'Accuracy']).sort_values('Accuracy', ascending=False)
print(results_df.to_string(index=False))
results_df.to_csv('outputs/model_comparison.csv', index=False)

# --- Bar chart of accuracies ---
plt.figure(figsize=(9, 5))
colors = ['#6c9ef8', '#e05c5c', '#6aeadb', '#f5a623']
sns.barplot(data=results_df, x='Accuracy', y='Model', palette=colors[:len(results_df)])
plt.xlim(0, 1)
plt.title('Model Comparison: Accuracy by Modality Strategy')
for i, v in enumerate(results_df['Accuracy']):
    plt.text(v + 0.01, i, f"{v:.3f}", va='center')
plt.tight_layout()
plt.savefig('outputs/01_model_comparison.png', dpi=120)
plt.close()
print("\nSaved: outputs/01_model_comparison.png")

# --- Training curves for the NumPy MLP ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(history['train_loss'], label='Train Loss', color='#f5a623')
axes[0].plot(history['val_loss'], label='Val Loss', color='#6c9ef8')
axes[0].set_title('Multi-Modal MLP — Loss Curve')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Cross-Entropy Loss')
axes[0].legend()

axes[1].plot(history['train_acc'], label='Train Accuracy', color='#f5a623')
axes[1].plot(history['val_acc'], label='Val Accuracy', color='#6c9ef8')
axes[1].set_title('Multi-Modal MLP — Accuracy Curve')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
axes[1].legend()

plt.tight_layout()
plt.savefig('outputs/02_training_curves.png', dpi=120)
plt.close()
print("Saved: outputs/02_training_curves.png")

# --- Confusion matrices for each model ---
fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
for ax, (name, preds) in zip(axes, predictions.items()):
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_, ax=ax, cbar=False)
    ax.set_title(f"{name}\nAcc={results[name]:.3f}", fontsize=10)
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig('outputs/03_confusion_matrices.png', dpi=120)
plt.close()
print("Saved: outputs/03_confusion_matrices.png")


# =================================================================
# 7. ABLATION: PERFORMANCE ON "DEGRADED" vs "CLEAN" SAMPLES
#    This is the key insight of the project: does fusion help most
#    on examples where one modality carries no useful signal
#    (occluded image OR generic text)?
# =================================================================
print("\n" + "=" * 60)
print("STEP 7: Ablation — Clean vs Degraded Samples")
print("=" * 60)

test_meta = df.iloc[idx_test].reset_index(drop=True)
any_degraded = (test_meta['image_occluded'] | test_meta['text_generic']).values
clean_mask = ~any_degraded

ablation_rows = []
for name, preds in predictions.items():
    acc_clean = accuracy_score(y_test[clean_mask], np.array(preds)[clean_mask])
    acc_degraded = accuracy_score(y_test[any_degraded], np.array(preds)[any_degraded]) if any_degraded.sum() > 0 else np.nan
    ablation_rows.append({'Model': name, 'Clean Samples': acc_clean, 'Degraded Samples (image OR text missing signal)': acc_degraded})

ablation_df = pd.DataFrame(ablation_rows)
print(ablation_df.to_string(index=False))
ablation_df.to_csv('outputs/ablation_clean_vs_degraded.csv', index=False)

# Plot
ablation_melt = ablation_df.melt(id_vars='Model', var_name='Subset', value_name='Accuracy')
plt.figure(figsize=(11, 5))
sns.barplot(data=ablation_melt, x='Model', y='Accuracy', hue='Subset', palette=['#6aeadb', '#e05c5c'])
plt.title('Accuracy on Clean vs Degraded (One Modality Uninformative) Samples')
plt.xticks(rotation=15, ha='right')
plt.ylim(0, 1.05)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.savefig('outputs/04_ablation_clean_vs_degraded.png', dpi=120)
plt.close()
print(f"\nDegraded samples in test set: {any_degraded.sum()} / {len(any_degraded)}")
print("Saved: outputs/04_ablation_clean_vs_degraded.png")


# =================================================================
# 8. SAVE EVERYTHING FOR INFERENCE
# =================================================================
print("\n" + "=" * 60)
print("STEP 8: Saving Models & Feature Extractors")
print("=" * 60)

joblib.dump(img_model, 'models/image_model.pkl')
joblib.dump(txt_model, 'models/text_model.pkl')
joblib.dump(fusion_rf, 'models/fusion_rf_model.pkl')
joblib.dump({'vectorizer': vectorizer, 'svd': svd,
              'img_scaler': img_scaler, 'txt_scaler': txt_scaler,
              'label_encoder': le}, 'models/preprocessors.pkl')
mlp.save('models/multimodal_mlp.npz')

print("Saved: models/image_model.pkl")
print("Saved: models/text_model.pkl")
print("Saved: models/fusion_rf_model.pkl")
print("Saved: models/multimodal_mlp.npz")
print("Saved: models/preprocessors.pkl")


# =================================================================
# 9. DETAILED CLASSIFICATION REPORT FOR BEST MODEL
# =================================================================
print("\n" + "=" * 60)
print("STEP 9: Classification Report — Multi-Modal Neural Net")
print("=" * 60)
print(classification_report(y_test, mlp_preds, target_names=le.classes_))

print("\n" + "=" * 60)
print("✅ PIPELINE COMPLETE")
print("=" * 60)
print(f"""
Summary:
  - Dataset:         {len(df)} samples, {len(le.classes_)} categories
  - Image features:  {X_img_train.shape[1]} dims (color histogram + HOG)
  - Text features:   {X_txt_train.shape[1]} dims (TF-IDF + SVD)

  Accuracy by approach:
""" + "\n".join(f"    - {name:35s}: {acc:.3f}" for name, acc in results.items()) + f"""

  Best model: {results_df.iloc[0]['Model']} ({results_df.iloc[0]['Accuracy']:.3f})
""")
