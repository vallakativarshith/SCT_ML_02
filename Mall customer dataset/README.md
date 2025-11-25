**Overview**
- **What**: K-means clustering script to segment retail customers by numeric features (purchase history proxies).
- **Script**: `cluster_customers.py` — reads CSV, scales features, produces elbow plot, fits KMeans, outputs cluster assignments and PCA plot.

**Quick Setup (Windows PowerShell)**
- Create and activate a Python environment (optional but recommended):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

- Install dependencies:

```powershell
pip install -r requirements.txt
```

**Basic usage**
- Run with defaults (expects `Mall_Customers.csv` in current folder):

```powershell
python .\cluster_customers.py
```

- Specify number of clusters and output file:

```powershell
python .\cluster_customers.py --k 4 --output clusters.csv
```

- Use specific columns (comma-separated):

```powershell
python .\cluster_customers.py --columns "Annual Income (k$),Spending Score (1-100)" --k 5
```

**Outputs**
- `clusters.csv` (default): input rows plus a `cluster` column
- `elbow.png` (default): inertia vs k plot to help choose k
- `clusters_pca.png` (default): 2D PCA visualization of clusters

**Notes & tips**
- If you don't pass `--columns`, numeric columns are inferred automatically (excluding common ID columns).
- If your dataset is large, increase `--max-k` for a broader elbow search, but plotting may take longer.
- Use the silhouette score printed by the script to evaluate cluster quality (higher is better).

If you want, I can run the script now on `Mall_Customers.csv` and produce the outputs. Do you want me to run it? (I'll need permission to run Python in this workspace.)
