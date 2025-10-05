"""
==================================================================
Simple Working Exoplanet Model - 6 Features
==================================================================
Creates a model that matches the API's 6-feature expectations.
This is the "old working model" configuration.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import json
import os
import warnings

warnings.filterwarnings('ignore')

print("="*70)
print("CREATING SIMPLE WORKING MODEL")
print("="*70)

# =============================================
# LOAD DATA
# =============================================
print("\n📂 Loading data...")
data_files = {
    'kepler': 'data/data.csv',
    'toi': 'data/TOI_2025.10.02_07.59.57.csv',
    'k2': 'data/k2pandc_2025.10.02_07.59.55.csv'
}

datasets = []
for name, path in data_files.items():
    try:
        df = pd.read_csv(path, comment='#')
        datasets.append(df)
        print(f"  ✓ Loaded {name}: {len(df)} samples")
    except Exception as e:
        print(f"  ⚠️  Skipped {name}: {e}")

combined_df = pd.concat(datasets, ignore_index=True)
print(f"\n✅ Total samples: {len(combined_df)}")

# =============================================
# CREATE 6 SIMPLE FEATURES (matching API)
# =============================================
print("\n🔧 Creating 6 features matching API...")

features = {
    'orbital_period': combined_df.get('koi_period', combined_df.get('pl_orbper', 1.0)),
    'planet_radius': combined_df.get('koi_prad', combined_df.get('pl_rade', 1.0)),
    'transit_depth': combined_df.get('koi_depth', combined_df.get('pl_trandep', 0.0)) / 1000000,
    'transit_duration': combined_df.get('koi_duration', combined_df.get('pl_trandurh', 1.0)),
    'insolation_flux': combined_df.get('koi_insol', combined_df.get('pl_insol', 1.0)),
    'equilibrium_temp': combined_df.get('koi_teq', combined_df.get('pl_eqt', 300.0))
}

X = pd.DataFrame(features)

# Handle target labels
if 'koi_disposition' in combined_df.columns:
    disposition_map = {
        'CONFIRMED': 'CONFIRMED',
        'CANDIDATE': 'CANDIDATE',
        'FALSE POSITIVE': 'FALSE_POSITIVE'
    }
    y = combined_df['koi_disposition'].map(disposition_map).fillna('CANDIDATE')
elif 'tfopwg_disp' in combined_df.columns:
    toi_map = {
        'CP': 'CONFIRMED', 'KP': 'CONFIRMED',
        'PC': 'CANDIDATE', 'APC': 'CANDIDATE'
    }
    y = combined_df['tfopwg_disp'].map(toi_map).fillna('CANDIDATE')
else:
    y = pd.Series(['CANDIDATE'] * len(combined_df))

# Filter valid labels
valid_labels = ['CONFIRMED', 'CANDIDATE', 'FALSE_POSITIVE']
mask = y.isin(valid_labels)
X = X[mask]
y = y[mask]

print(f"✅ Features: {list(X.columns)}")
print(f"✅ Samples: {len(X)}")
print(f"✅ Class distribution:")
for label, count in y.value_counts().items():
    print(f"   {label}: {count}")

# =============================================
# HANDLE MISSING VALUES
# =============================================
print("\n🔄 Handling missing values...")
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(
    imputer.fit_transform(X),
    columns=X.columns,
    index=X.index
)

# =============================================
# TRAIN/TEST SPLIT
# =============================================
print("\n✂️  Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Training: {len(X_train)} samples")
print(f"  Testing:  {len(X_test)} samples")

# =============================================
# SCALE FEATURES
# =============================================
print("\n📏 Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =============================================
# TRAIN MODEL (Simple, Reliable Configuration)
# =============================================
print("\n🤖 Training Gradient Boosting model...")
print("   Parameters: n_estimators=100, learning_rate=0.1, max_depth=5")

model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    verbose=0
)

model.fit(X_train_scaled, y_train)
print("✅ Training complete!")

# =============================================
# EVALUATE MODEL
# =============================================
print("\n📊 Evaluating model...")
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"\n{'='*70}")
print(f"MODEL PERFORMANCE")
print(f"{'='*70}")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred))

# =============================================
# SAVE MODEL AND METRICS
# =============================================
print("\n💾 Saving model and metrics...")

os.makedirs('models', exist_ok=True)
os.makedirs('metrics', exist_ok=True)

# Save as pipeline format matching utils.py expectations
pipeline = {
    'model': model,
    'scaler': scaler,
    'imputer': imputer,
    'feature_names': list(X.columns),
    'target_classes': list(model.classes_)
}

joblib.dump(pipeline, 'models/pipeline.joblib')
print("  ✓ Model saved: models/pipeline.joblib")

# Save metrics
metrics = {
    'accuracy': float(accuracy),
    'f1_score': float(f1),
    'test_samples': int(len(y_test)),
    'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
    'classes': list(model.classes_),
    'classification_report': classification_report(y_test, y_pred, output_dict=True),
    'model_params': model.get_params(),
    'feature_names': list(X.columns)
}

with open('metrics/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
print("  ✓ Metrics saved: metrics/metrics.json")

print("\n" + "="*70)
print("✅ MODEL READY FOR USE!")
print("="*70)
print("\nThis simple 6-feature model matches your API expectations.")
print("You can now start the API server: python api.py")
