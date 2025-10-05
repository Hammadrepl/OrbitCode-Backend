# 🚀 Exoplanet AI Classifier

[![NASA Space Apps Challenge](https://img.shields.io/badge/NASA-Space%20Apps%20Challenge-blue)](https://www.spaceappschallenge.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Accuracy](https://img.shields.io/badge/Accuracy-91.6%25-success)](metrics/metrics.json)

Physics-informed AI classifier for exoplanet candidates using NASA Kepler data. Achieves 91.6% accuracy through gradient boosting + astronomical physics rules.

## Quick Start

```bash
# Install
git clone <repo-url>
cd backend
pip install -r requirements.txt

# Train model
python train_model.py

# Start API server
python api.py
```

**API Usage:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"orbital_period": 3.52, "planet_radius": 15.5, "transit_depth": 0.014}'
```

**Python Usage:**
```python
from utils import predict_single
result = predict_single(orbital_period=3.52, planet_radius=15.5, transit_depth=0.014)
print(result['prediction'])  # CONFIRMED
```

## API Endpoints

- `POST /predict` - Single prediction
- `GET /health` - Health check
- `GET /model_info` - Model metadata

## Files

- `train_model.py` - Training script
- `utils.py` - Prediction functions with physics rules
- `api.py` - REST API server
- `requirements.txt` - Dependencies

---

**Built for NASA Space Apps Challenge** 🚀
