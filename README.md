### Jungle Route Profiler ###

This project clusters and visualizes jungle paths of players from League of Legends using unsupervised learning on spatial-temporal footprint data.

## Contents

- `psy_jungle_300.csv`: A dataset of jungle path coordinates across multiple players and time intervals.
- `map11.png`: A stylized background map image used to overlay and visualize the jungle routes.
- `route_profile_generator_psy.py`: Python script to process, cluster, and visualize player jungle routes over discrete time windows.

## Overview

The Python script performs the following steps:

1. **Data Loading**: Reads player coordinate data from the CSV file.
2. **Time Segmentation**: Splits the match timeline into five 10-second windows (e.g., 0–5 min, 5–10 min, ...).
3. **Feature Extraction**: Extracts `x` and `y` coordinate sequences for each time segment.
4. **Clustering**: Applies KMeans clustering to group similar jungle paths in each time window.
5. **Visualization**: Uses `matplotlib` to overlay player routes on `map11.png`, colored by cluster label.

## Requirements

Install dependencies with:

```bash
pip install pandas matplotlib scikit-learn opencv-python numpy
