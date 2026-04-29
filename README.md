# Recommendation System: IBM Watson Studio Community

## Project Overview

This project analyzes user interactions with articles on the IBM Watson Studio platform and builds a multi-method recommendation engine. It was completed as part of the **Udacity Data Science Nanodegree** program.

Four distinct recommendation strategies are implemented, ranging from simple popularity-based ranking to SVD-based matrix factorization, allowing the system to handle a variety of user scenarios — including brand new users with no interaction history.

---

## Project Structure

```
starter/
├── Recommendations_with_IBM.ipynb   # Main project notebook
├── project_tests.py                 # Automated grading test suite
└── data/
    ├── user-item-interactions.csv   # User-article interaction logs (user_id, article_id, title)
    └── articles_community.csv       # Article metadata provided by Udacity (not used in this project)
```

---

## Dataset

The interaction dataset contains `user_id`, `article_id`, and `title` columns. Key statistics:

- **5,149** unique users
- **714** unique articles with at least one interaction
- The most interacted-with article had **937** views
- The median number of articles per user is **3**
- Missing `email` values (null users) are assigned a shared `"unknown_user"` identifier

---

## Methods Implemented

### Part I — Exploratory Data Analysis
Explores the dataset to understand interaction distributions, identifies null values, and computes descriptive statistics. Users are mapped from email addresses to integer IDs via `email_mapper()`.

### Part II — Rank-Based Recommendations
Recommends the most globally popular articles by total interaction count. Two functions are implemented:

- `get_top_articles(n)` — returns top `n` article titles
- `get_top_article_ids(n)` — returns top `n` article IDs

This method is used as the **cold start fallback** for new users with no history.

### Part III — User-User Collaborative Filtering
Builds a binary user-item matrix (5,149 users × 714 articles) and finds similar users using **cosine similarity**.

Key functions:
- `create_user_item_matrix(df)` — creates the binary interaction matrix
- `find_similar_users(user_id)` — returns users ordered by cosine similarity
- `get_top_sorted_users(user_id)` — returns a DataFrame of neighbors sorted by cosine similarity, then by number of total interactions to break ties
- `user_user_recs(user_id, m)` — basic collaborative filtering recommendations
- `user_user_recs_part2(user_id, m)` — improved version that collects all candidate articles first, then ranks them by global popularity before returning the top `m`

### Part IV — Content-Based Recommendations
Uses **TF-IDF** on article titles combined with **LSA (Latent Semantic Analysis)** via TruncatedSVD, followed by **KMeans clustering** (50 clusters) to group articles by content similarity.

Key functions:
- `get_similar_articles(article_id)` — returns all articles in the same KMeans cluster
- `make_content_recs(article_id, n)` — returns the top `n` similar articles ranked by interaction popularity

TF-IDF settings: `max_features=200`, `max_df=0.75`, `min_df=5`, English stop words removed. LSA uses 50 components.

### Part V — Matrix Factorization (SVD)
Applies **TruncatedSVD** from scikit-learn to the full user-item matrix to learn latent user and article features.

- **U** (users × latent features), **S** (singular values), **Vt** (latent features × articles) are extracted
- Accuracy, precision, and recall are tracked across 10–700 latent features
- **150–200 latent features** is selected as the optimal point where recall gains begin to flatten, balancing expressiveness with generalization risk
- `get_svd_similar_article_ids(article_id, vt)` — finds similar articles using **cosine similarity** on article embeddings from Vt

---

## Key Design Decisions

**Cosine similarity vs. dot product for user similarity**  
Cosine similarity is used in both `find_similar_users` and `get_top_sorted_users` because it normalizes for user activity level — highly active users don't dominate similarity scores simply by having interacted with more articles. This produces fairer, more meaningful comparisons across users with different levels of engagement.

**Collecting all candidates before ranking in `user_user_recs_part2`**  
Gathering all candidate articles into a set first, then ranking by global popularity, ensures the final recommendations reflect the most consistently engaging content rather than the arbitrary order in which similar users were traversed.

**50 KMeans clusters for content-based recommendations**  
An elbow plot of KMeans inertia across cluster counts showed a clear inflection around 50 clusters, making it the natural choice for grouping the ~714 unique articles.

**Cold start handling**  
New users with zero interaction history cannot receive collaborative filtering or SVD-based recommendations. The system falls back to rank-based recommendations (`get_top_article_ids`), which requires no user history.

---

## Recommendation Strategy by User Type

| User Type | Recommended Method |
|---|---|
| New user (no history) | Rank-based (Part II) |
| User with 1–2 interactions | Content-based (Part IV) |
| User with moderate history | User-user collaborative filtering (Part III) |
| Active user with rich history | Matrix factorization / SVD (Part V) |

A hybrid approach — starting with popularity, transitioning through content-based, and ultimately leveraging collaborative filtering or SVD — is the most robust strategy in practice.

---

## Evaluation Discussion

Offline metrics (accuracy, precision, recall) on the binary user-item matrix are limited as evaluation tools: because the matrix is highly sparse, predicting zeros is trivially easy and inflates accuracy regardless of recommendation quality.

A more meaningful evaluation would require **online A/B testing** — splitting users across recommendation strategies and measuring real engagement metrics such as click-through rate, session depth, or return visits.

---

## Libraries Used

- `pandas`, `numpy` — data manipulation and matrix operations
- `scikit-learn` — cosine similarity, TF-IDF, TruncatedSVD, KMeans, precision/recall/accuracy metrics
- `matplotlib` — visualizations

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/Eazi-T/dsnd-recommendation-systems.git
   cd dsnd-recommendation-systems/starter
   ```

2. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib jupyter
   ```

3. Launch the notebook:
   ```bash
   jupyter notebook Recommendations_with_IBM.ipynb
   ```

4. Run all cells in order. All automated tests should pass.

---

## Acknowledgements

- Dataset and project framework provided by [IBM Watson Studio](https://www.ibm.com/cloud/watson-studio) via [Udacity](https://www.udacity.com/)
- Completed as part of the **Udacity Data Science Nanodegree**

## License

[License](LICENSE.txt)
