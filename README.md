# Task-2 · Personalized Restaurant Recommendation Engine

<p>
  <img src="https://img.shields.io/badge/Type-Content--Based%20Filtering-4A90D9?style=flat-square" />
  <img src="https://img.shields.io/badge/Corpus-9%2C551%20restaurants-FF6B35?style=flat-square" />
  <img src="https://img.shields.io/badge/Cities-141-2ECC71?style=flat-square" />
  <img src="https://img.shields.io/badge/Validation%20Scenarios-3-lightgrey?style=flat-square" />
</p>

---

## Overview

A content-based restaurant recommendation engine that scores and ranks the full restaurant corpus against a user's stated preferences — cuisine type, budget tier, minimum acceptable rating, and preferred city. The system requires no prior interaction history and operates entirely on restaurant feature attributes, making it immediately deployable for new users and cold-start environments.

---

## Problem Statement

Collaborative filtering models depend on accumulated user interaction data, which is unavailable for new users or recently launched platforms. A content-based approach removes this dependency entirely: recommendations are derived from the intrinsic attributes of each restaurant, delivering relevant results from the first query.

---

## Objective

Design and implement a weighted content-based filtering algorithm that accepts four user preference parameters and returns the top-N best-matched restaurants with fully interpretable, normalized match scores.

---

## Technologies Used

| Category | Tool |
|----------|------|
| Language | Python 3.9+ |
| Data Manipulation | pandas, numpy |
| Visualization | matplotlib, seaborn |
| Environment | Jupyter Notebook |

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
| Total Restaurants | 9,551 |
| Rated Restaurants (used) | 7,403 · 0.0-rated records excluded |
| Cities | 141 |
| Unique Cuisine Combinations | 1,825+ |
| Price Range Tiers | 1 (Budget) → 4 (Premium) |
| Active Rating Range | 0.5 – 4.9 |

---

## Project Workflow

```
Raw Dataset  ·  9,551 rows
        │
        ▼
 01  Load and filter
     Remove 0.0-rated records  →  7,403 restaurants retained
        │
        ▼
 02  Define user preference parameters
     cuisine_pref  ·  budget_tier  ·  min_rating  ·  city_pref  ·  top_n
        │
        ▼
 03  Compute composite match score for every restaurant
     using the weighted scoring formula
        │
        ▼
 04  Disqualify restaurants below min_rating
     (score forced to -1 · excluded from ranking)
        │
        ▼
 05  Sort by match score descending  →  return top-N results
        │
        ▼
 06  Validate output across 3 real-world test scenarios
```

---

## Machine Learning Techniques Used

### Algorithm · Weighted Content-Based Filtering

Every restaurant in the corpus receives a composite match score bounded to [0.0, 1.0]:

```
match_score  =  0.35 × cuisine_score
              + 0.30 × rating_score
              + 0.20 × budget_score
              + 0.15 × city_score
```

### Scoring Logic

| Criterion | Weight | Scoring Rule |
|-----------|--------|--------------|
| Cuisine | 35% | 1.0 if the preferred cuisine appears anywhere in the restaurant's cuisine string · 0.0 otherwise |
| Rating | 30% | `rating ÷ 4.9` (normalized to 0–1) · restaurants below `min_rating` are excluded entirely |
| Budget | 20% | 1.0 exact tier match · 0.5 adjacent tier · 0.0 more than one tier away |
| City | 15% | 1.0 city matches preference · 0.0 city differs (restaurant remains eligible) |

### Weight Rationale

Cuisine carries the highest weight (35%) because it represents the most personal and non-negotiable user preference. Rating (30%) enforces a quality floor. Budget (20%) reflects a practical financial constraint. City is deliberately soft-weighted (15%) to allow high-quality out-of-city restaurants to surface — supporting discovery over strict locality filtering.

---

## Results

### Scenario Validation

| Scenario | User Preferences | Top Recommended Result | Match Score |
|----------|-----------------|------------------------|-------------|
| 1 | North Indian · Tier 2 · New Delhi · ≥ 3.5 | Food Scouts (rating 4.6) | 0.982 |
| 2 | Italian · Tier 4 · Any city · ≥ 4.0 | Zolocrust – Hotel Clarks Amer (rating 4.9) | 1.000 |
| 3 | Chinese · Tier 1 · Mumbai · ≥ 3.0 | Sheroes Hangout (rating 4.9) | 0.850 |

All three scenarios returned contextually appropriate, high-quality recommendations. Match scores remained within the [0.0, 1.0] bound across the entire corpus.

### Qualitative Evaluation

| Quality Check | Outcome |
|---------------|---------|
| All top results meet minimum rating threshold | ✓ Pass |
| Cuisine matching via substring search applied correctly | ✓ Pass |
| Budget adjacency logic executes correctly | ✓ Pass |
| Match scores bounded to [0.0, 1.0] | ✓ Pass |
| All 3 test scenarios returned relevant recommendations | ✓ Pass |

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

**Execute the script:**

```bash
python src/recommendation_engine.py
```

**Call the `recommend()` function directly:**

```python
from src.recommendation_engine import recommend

results = recommend(
    cuisine_pref='North Indian',
    budget_tier=2,       # 1 = Budget  ·  2 = Moderate  ·  3 = Expensive  ·  4 = Premium
    min_rating=3.5,
    city_pref='New Delhi',
    top_n=10
)
print(results)
```

**Launch the notebook:**

```bash
jupyter notebook notebook.ipynb
```

---

## Future Improvements

- Introduce a collaborative filtering layer driven by implicit user interaction signals
- Replace the binary city match with Haversine distance scoring using the existing latitude/longitude columns
- Apply TF-IDF vectorization to cuisine strings to enable soft, partial-match scoring
- Integrate approximate nearest-neighbour indexing (Faiss, Annoy) for large-scale retrieval
- Expose the engine via a Streamlit or FastAPI interface for interactive use

---

## Author

**Abhishek** · Ref: CTI/A1/C358755
BCA · Amity University Online, Noida
Machine Learning Internship · Cognifyz Technologies

---
---
