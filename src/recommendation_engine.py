# ============================================================
# Task 2: Personalized Restaurant Recommendation Engine
# Intern: Abhishek | Enrollment: CTI/A1/C358755
# Organization: Cognifyz Technologies
# Course: BCA | Domain: Machine Learning
# ============================================================
#
# APPROACH: Content-Based Filtering
# We build a preference profile from user inputs (cuisine,
# budget tier, min rating, city) and score every restaurant
# in the dataset against that profile using a weighted
# similarity formula — no external libraries required.
# ============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

SCREENSHOTS = os.path.join(os.path.dirname(__file__), '..', 'Screenshots')
os.makedirs(SCREENSHOTS, exist_ok=True)

# ── 1. Load & Clean ────────────────────────────────────────
df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'Dataset', 'Dataset.csv'))

# Fill missing cuisines
df['Cuisines'] = df['Cuisines'].fillna('Unknown')

# Remove unrated restaurants (0.0 = Not Rated in this dataset)
df = df[df['Aggregate rating'] > 0].copy()
df.reset_index(drop=True, inplace=True)

# Normalise string columns for safe comparison
df['City_clean']    = df['City'].str.strip().str.lower()
df['Cuisines_clean'] = df['Cuisines'].str.lower()

print(f"Working dataset: {df.shape[0]} rated restaurants across {df['City'].nunique()} cities.")

# ── 2. Scoring Logic ───────────────────────────────────────
def compute_score(row, cuisine_pref, budget_tier, min_rating, city_pref, weights):
    """
    Score a restaurant row against user preferences.

    Scoring components (each 0–1):
      cuisine_score  : 1 if preferred cuisine appears in restaurant cuisines, else 0
      budget_score   : 1 if price range matches user budget tier, else partial
      rating_score   : normalised rating (0–4.9 → 0–1)
      city_score     : 1 if city matches, else 0

    Final score = weighted sum of above components.
    """
    # Cuisine match
    cuisine_score = 1.0 if cuisine_pref.lower() in row['Cuisines_clean'] else 0.0

    # Budget match: price range 1–4; give full mark if exact, 0.5 if adjacent
    diff = abs(row['Price range'] - budget_tier)
    if diff == 0:
        budget_score = 1.0
    elif diff == 1:
        budget_score = 0.5
    else:
        budget_score = 0.0

    # Rating: only consider restaurants that meet minimum threshold
    if row['Aggregate rating'] < min_rating:
        return -1  # disqualified
    rating_score = row['Aggregate rating'] / 4.9  # normalise to 0–1

    # City match
    city_score = 1.0 if city_pref.lower() in row['City_clean'] else 0.0

    score = (weights['cuisine'] * cuisine_score +
             weights['budget']  * budget_score  +
             weights['rating']  * rating_score  +
             weights['city']    * city_score)
    return score


def recommend(cuisine_pref='North Indian', budget_tier=2, min_rating=3.5,
              city_pref='New Delhi', top_n=10,
              weights=None):
    """
    Return top-N restaurant recommendations based on user preferences.

    Parameters
    ----------
    cuisine_pref : str   – cuisine type the user enjoys
    budget_tier  : int   – 1 (cheap) to 4 (expensive)
    min_rating   : float – minimum acceptable aggregate rating
    city_pref    : str   – preferred city
    top_n        : int   – number of results to return
    weights      : dict  – importance of each criterion
    """
    if weights is None:
        weights = {'cuisine': 0.35, 'budget': 0.20, 'rating': 0.30, 'city': 0.15}

    scores = df.apply(
        lambda row: compute_score(row, cuisine_pref, budget_tier, min_rating, city_pref, weights),
        axis=1
    )

    # Filter out disqualified rows
    mask = scores >= 0
    scored_df = df[mask].copy()
    scored_df['match_score'] = scores[mask].values

    if scored_df.empty:
        print("No restaurants match the given preferences. Try relaxing the filters.")
        return pd.DataFrame()

    top = scored_df.sort_values('match_score', ascending=False).head(top_n)

    output_cols = [
        'Restaurant Name', 'City', 'Cuisines',
        'Average Cost for two', 'Price range',
        'Aggregate rating', 'Votes', 'match_score'
    ]
    return top[output_cols].reset_index(drop=True)


# ── 3. Run Sample Scenarios ────────────────────────────────
print("\n" + "=" * 60)
print("SCENARIO 1: North Indian food, mid budget, Delhi, rating ≥ 3.5")
print("=" * 60)
s1 = recommend(cuisine_pref='North Indian', budget_tier=2, min_rating=3.5,
               city_pref='New Delhi', top_n=5)
print(s1.to_string(index=False))

print("\n" + "=" * 60)
print("SCENARIO 2: Italian cuisine, premium budget, any city, rating ≥ 4.0")
print("=" * 60)
s2 = recommend(cuisine_pref='Italian', budget_tier=4, min_rating=4.0,
               city_pref='', top_n=5,
               weights={'cuisine': 0.50, 'budget': 0.20, 'rating': 0.30, 'city': 0.0})
print(s2.to_string(index=False))

print("\n" + "=" * 60)
print("SCENARIO 3: Chinese food, cheapest budget, Mumbai, rating ≥ 3.0")
print("=" * 60)
s3 = recommend(cuisine_pref='Chinese', budget_tier=1, min_rating=3.0,
               city_pref='Mumbai', top_n=5)
print(s3.to_string(index=False))

# ── 4. Visualizations ─────────────────────────────────────
sns.set_theme(style='whitegrid')

# 4a. Top cuisines in the dataset
df['Primary Cuisine'] = df['Cuisines'].apply(lambda x: x.split(',')[0].strip())
top_cuisines = df['Primary Cuisine'].value_counts().head(15)
fig, ax = plt.subplots(figsize=(10, 6))
top_cuisines.sort_values().plot(kind='barh', ax=ax, color='coral', edgecolor='white')
ax.set_title('Top 15 Primary Cuisines in Dataset', fontsize=14, fontweight='bold')
ax.set_xlabel('Number of Restaurants')
fig.tight_layout()
fig.savefig(os.path.join(SCREENSHOTS, '01_top_cuisines.png'), dpi=150)
plt.close(fig)
print("\nSaved: 01_top_cuisines.png")

# 4b. Price range distribution
fig, ax = plt.subplots(figsize=(7, 5))
df['Price range'].value_counts().sort_index().plot(kind='bar', ax=ax,
    color=['#4CAF50','#2196F3','#FF9800','#F44336'], edgecolor='white', rot=0)
ax.set_title('Restaurant Distribution by Price Range (1=Cheap, 4=Expensive)',
             fontsize=12, fontweight='bold')
ax.set_xlabel('Price Range')
ax.set_ylabel('Count')
fig.tight_layout()
fig.savefig(os.path.join(SCREENSHOTS, '02_price_range_distribution.png'), dpi=150)
plt.close(fig)
print("Saved: 02_price_range_distribution.png")

# 4c. Rating distribution by price tier
fig, ax = plt.subplots(figsize=(9, 5))
for tier, color in zip([1, 2, 3, 4], ['#4CAF50','#2196F3','#FF9800','#F44336']):
    subset = df[df['Price range'] == tier]['Aggregate rating']
    ax.hist(subset, bins=15, alpha=0.6, label=f'Price Range {tier}', color=color, edgecolor='none')
ax.set_title('Rating Distribution by Price Range', fontsize=13, fontweight='bold')
ax.set_xlabel('Aggregate Rating')
ax.set_ylabel('Count')
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(SCREENSHOTS, '03_rating_by_price_range.png'), dpi=150)
plt.close(fig)
print("Saved: 03_rating_by_price_range.png")

# 4d. Online delivery vs average rating
fig, ax = plt.subplots(figsize=(7, 5))
df.groupby('Has Online delivery')['Aggregate rating'].mean().plot(
    kind='bar', ax=ax, color=['#e07b54', '#5b8db8'], edgecolor='white', rot=0, width=0.5)
ax.set_title('Average Rating: With vs Without Online Delivery', fontsize=12, fontweight='bold')
ax.set_xlabel('Has Online Delivery')
ax.set_ylabel('Average Rating')
ax.set_ylim(0, 5)
fig.tight_layout()
fig.savefig(os.path.join(SCREENSHOTS, '04_delivery_vs_rating.png'), dpi=150)
plt.close(fig)
print("Saved: 04_delivery_vs_rating.png")

print("\n✅ Task 2 Complete — Recommendation Engine Ready.")
