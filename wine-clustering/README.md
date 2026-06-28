# Wine Cultivar Clustering

Unsupervised learning project applying K-Means and Hierarchical Clustering to the sklearn Wine 
dataset, with PCA and t-SNE for dimensionality reduction.

## Contents
- `Wejdan_wine_clustering.ipynb` — main analysis notebook
- `figures/` — saved plots (Silhouette scores, PCA comparison, t-SNE)
- `report/` — 2-page PDF and DOCX summary report

## Summary
K-Means and Hierarchical Clustering both identified 3 natural clusters (k=3), matching the dataset's 
true cultivar count without using any label information. K-Means achieved a Silhouette score of 0.2849 
and recovered the true cultivar structure with 96.6% agreement.
