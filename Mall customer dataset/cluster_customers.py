#!/usr/bin/env python3
"""
cluster_customers.py

Simple, flexible K-means clustering script for customer segmentation.
Reads a CSV (default `Mall_Customers.csv`), selects numeric columns (or user-specified),
scales features, computes an elbow plot, fits KMeans, saves cluster assignments,
and writes PCA-based 2D cluster visualization.

Usage examples in README.md.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")


def infer_numeric_columns(df: pd.DataFrame):
    exclude_names = {"CustomerID", "Customer Id", "ID", "Id"}
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    num = [c for c in num if c not in exclude_names]
    return num


def plot_elbow(inertias, out_path: Path):
    plt.figure(figsize=(7, 4))
    ks = list(range(1, len(inertias) + 1))
    plt.plot(ks, inertias, '-o')
    plt.xticks(ks)
    plt.xlabel('k (number of clusters)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method: inertia vs k')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_pca_clusters(X_scaled, labels, out_path: Path, palette='tab10'):
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_scaled)
    dfp = pd.DataFrame(coords, columns=['PC1', 'PC2'])
    dfp['cluster'] = labels.astype(str)

    plt.figure(figsize=(7, 6))
    sns.scatterplot(data=dfp, x='PC1', y='PC2', hue='cluster', palette=palette, s=60)
    plt.title('Clusters visualized with PCA (2 components)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend(title='cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    p = argparse.ArgumentParser(description='K-means clustering for customer segmentation')
    p.add_argument('-i', '--input', default='Mall_Customers.csv', help='Input CSV file (default: Mall_Customers.csv)')
    p.add_argument('-c', '--columns', help='Comma-separated list of columns to use for clustering (default: infer numeric columns)')
    p.add_argument('-k', '--k', type=int, default=3, help='Number of clusters (default: 3)')
    p.add_argument('--max-k', type=int, default=10, help='Max k to try for elbow plot (default: 10)')
    p.add_argument('-o', '--output', default='clusters.csv', help='Output CSV with cluster labels (default: clusters.csv)')
    p.add_argument('--elbow-out', default='elbow.png', help='Elbow plot output path')
    p.add_argument('--pca-out', default='clusters_pca.png', help='PCA cluster plot output path')
    p.add_argument('--random-seed', type=int, default=42, help='Random seed for reproducibility')
    args = p.parse_args()

    infile = Path(args.input)
    if not infile.exists():
        print(f"Input file not found: {infile.resolve()}")
        sys.exit(2)

    df = pd.read_csv(infile)
    if args.columns:
        cols = [c.strip() for c in args.columns.split(',') if c.strip()]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            print(f"Columns not found in CSV: {missing}")
            sys.exit(3)
    else:
        cols = infer_numeric_columns(df)
        if not cols:
            print("No numeric columns inferred from CSV. Please pass `--columns`.")
            sys.exit(4)

    print(f"Using columns for clustering: {cols}")
    X = df[cols].copy()
    # Keep index alignment for putting cluster labels back
    idx = X.dropna().index
    X = X.loc[idx]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Elbow plot
    max_k = max(1, args.max_k)
    inertias = []
    for k in range(1, max_k + 1):
        model = KMeans(n_clusters=k, random_state=args.random_seed, n_init=10)
        model.fit(X_scaled)
        inertias.append(model.inertia_)

    try:
        plot_elbow(inertias, Path(args.elbow_out))
        print(f"Elbow plot written to {args.elbow_out}")
    except Exception as e:
        print(f"Warning: could not write elbow plot: {e}")

    if args.k < 1:
        print("k must be >= 1")
        sys.exit(5)

    model = KMeans(n_clusters=args.k, random_state=args.random_seed, n_init=10)
    model.fit(X_scaled)
    labels = model.labels_

    # Attach cluster labels to original dataframe
    out_df = df.copy()
    out_df.loc[idx, 'cluster'] = labels

    out_path = Path(args.output)
    out_df.to_csv(out_path, index=False)
    print(f"Clusters saved to {out_path}")

    if args.k > 1:
        try:
            sil = silhouette_score(X_scaled, labels)
            print(f"Silhouette score (k={args.k}): {sil:.4f}")
        except Exception:
            print("Silhouette score: calculation failed (small sample or identical points)")

    try:
        plot_pca_clusters(X_scaled, labels, Path(args.pca_out))
        print(f"PCA cluster plot written to {args.pca_out}")
    except Exception as e:
        print(f"Warning: could not write PCA plot: {e}")

    print("Done.")


if __name__ == '__main__':
    main()
