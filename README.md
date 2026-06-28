# Task-2 — Personalized Restaurant Recommendation Engine

<p>
  <img src="https://img.shields.io/badge/Type-Content--Based%20Filtering-4A90D9?style=flat-square" />
  <img src="https://img.shields.io/badge/Restaurants-9%2C551-FF6B35?style=flat-square" />
  <img src="https://img.shields.io/badge/Cities-141-2ECC71?style=flat-square" />
  <img src="https://img.shields.io/badge/Scenarios%20Validated-3-lightgrey?style=flat-square" />
</p>

---

## Overview

A content-based restaurant recommendation engine that scores and ranks restaurants based on a user's stated preferences — cuisine type, budget tier, minimum acceptable rating, and preferred city. The system operates without any user interaction history and is driven entirely by restaurant feature attributes.

---

## Problem Statement

Restaurant discovery platforms often require user interaction history to power collaborative filtering models. For new users or platforms with sparse interaction data, a content-based approach can surface relevant recommendations immediately using only restaurant characteristics.

---

## Objective

Implement a weighted content-based filtering algorithm that accepts four user preference parameters and returns the top-N best-matching restaurants, with transparent, interpretable scoring.

---

## Technologies Used

| Category | Tool |
|----------|------|
| Language | Python 3.9+ |
| Data manipulation | pandas, numpy |
| Visualization | matplotlib, seaborn |
| Notebook | Jupyter |

---

## Python Libraries

```python
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
jupyter>=1.0.0
```

---

## Dataset Information

| Property | Value |
|----------|-------|
| File | `dataset/Dataset.csv` |
| Total restaurants | 9,551 |
| Rated restaurants used | 7,403 (0.0-rated excluded) |
| Cities covered | 141 |
| Unique cuisine combinations | 1,825+ |
| Price range tiers | 1 (Cheap) to 4 (Premium) |
| Rating range (filtered) | 0.5 to 4.9 |

---

## Project Workflow

```
Raw Dataset (9,551 rows)
        │
        ▼
1. Load and filter — remove 0.0-rated restaurants (Not Rated)
        │
        ▼
2. Define user preference parameters:
   cuisine_pref, budget_tier, min_rating, city_pref, top_n
        │
        ▼
3. Compute match score for each restaurant using weighted formula
        │
        ▼
4. Disqualify restaurants below min_rating (score → -1)
        │
        ▼
5. Sort by score descending → return top-N results
        │
        ▼
6. Validate across 3 test scenarios
```

---

## Machine Learning Techniques Used

### Algorithm: Weighted Content-Based Filtering

Each restaurant receives a composite match score:

```
score = 0.35 × cuisine_score
      + 0.30 × rating_score
      + 0.20 × budget_score
      + 0.15 × city_score
```

**Scoring rules:**

| Criterion | Weight | Scoring Logic |
|-----------|--------|---------------|
| Cuisine match | 35% | 1.0 if preferred cuisine appears in restaurant's cuisines; 0.0 otherwise |
| Rating score | 30% | `rating / 4.9` (normalized 0–1); restaurants below `min_rating` are excluded entirely |
| Budget match | 20% | 1.0 (exact tier match) · 0.5 (adjacent tier) · 0.0 (far tier) |
| City match | 15% | 1.0 (city matches) · 0.0 (city mismatch, still eligible) |

**Weight rationale:**
- Cuisine carries the most weight (35%) as it is the most personal, non-negotiable preference
- Rating (30%) ensures quality assurance
- Budget (20%) is a practical hard constraint
- City (15%) is soft — excellent restaurants surface even on city mismatches, enabling discovery

---

## Results

### Scenario Test Outputs

| Scenario | Preferences | Top Result | Match Score |
|----------|-------------|------------|-------------|
| 1 | North Indian · Tier 2 · New Delhi · ≥ 3.5 | Food Scouts (rating 4.6) | 0.982 |
| 2 | Italian · Tier 4 · Any city · ≥ 4.0 | Zolocrust – Hotel Clarks Amer (rating 4.9) | 1.000 |
| 3 | Chinese · Tier 1 · Mumbai · ≥ 3.0 | Sheroes Hangout (rating 4.9) | 0.850 |

All scenarios returned contextually relevant, sensible recommendations. Match scores are bounded [0.0, 1.0].

### Qualitative Evaluation

| Check | Result |
|-------|--------|
| Top results always meet minimum rating threshold | ✓ Pass |
| Cuisine matching applied correctly (substring search) | ✓ Pass |
| Budget adjacency logic works | ✓ Pass |
| Match scores bounded 0.0 to 1.0 | ✓ Pass |
| All 3 test scenarios returned relevant results | ✓ Pass |

---

## Folder Structure

```
Task-2/
├── README.md
├── notebook.ipynb
├── src/
│   └── recommendation_engine.py
├── dataset/
│   └── Dataset.csv
├── outputs/
└── images/
    ├── 01_top_cuisines.png
    ├── 02_price_range_distribution.png
    ├── 03_rating_by_price_range.png
    └── 04_delivery_vs_rating.png
```

---

## Installation

```bash
cd Task-2
pip install -r ../requirements.txt
```

---

## Usage

**Run the Python script:**

```bash
python src/recommendation_engine.py
```

**Use the `recommend()` function directly:**

```python
from src.recommendation_engine import recommend

results = recommend(
    cuisine_pref='North Indian',
    budget_tier=2,        # 1=cheap · 2=moderate · 3=expensive · 4=premium
    min_rating=3.5,
    city_pref='New Delhi',
    top_n=10
)
print(results)
```

**Run the notebook:**

```bash
jupyter notebook notebook.ipynb
```

---

## Future Improvements

- Add collaborative filtering layer using implicit feedback data
- Replace binary city match with Haversine distance scoring (lat/lon already present)
- Vectorize cuisine descriptions with TF-IDF for partial match scoring
- Add approximate nearest-neighbour indexing (Faiss, Annoy) for large-scale retrieval
- Build an interactive web UI with Streamlit or FastAPI

---

## Author

**Abhishek** | Ref: CTI/A1/C358755
BCA — Amity University Online, Noida
Machine Learning Internship @ Cognifyz Technologies
