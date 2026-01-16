# core/visualizer.py
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class DataVisualizer:
    def __init__(self, plots_dir: str):
        self.plots_dir = plots_dir
        os.makedirs(self.plots_dir, exist_ok=True)

    def correlation_heatmap(self, df: pd.DataFrame, filename: str = "heatmap.png"):
        plt.figure(figsize=(10, 8))
        sns.heatmap(df.select_dtypes(include=["number"]).corr(), annot=False, cmap="viridis")
        plt.tight_layout()
        path = os.path.join(self.plots_dir, filename)
        plt.savefig(path)
        plt.close()
        return path

    def numeric_histograms(self, df: pd.DataFrame, max_plots: int = 6):
        saved = []
        num_cols = df.select_dtypes(include=["number"]).columns.tolist()[:max_plots]
        for c in num_cols:
            plt.figure()
            sns.histplot(df[c].dropna(), kde=True)
            plt.title(c)
            p = os.path.join(self.plots_dir, f"{c}_hist.png")
            plt.tight_layout()
            plt.savefig(p)
            plt.close()
            saved.append(p)
        return saved
