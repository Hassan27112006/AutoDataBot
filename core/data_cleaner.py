# core/data_cleaner.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from typing import Tuple

class DataCleaner:
    """
    Cleans incoming DataFrame:
    - Drops duplicates
    - Simple imputation: median for numeric, mode for categorical
    - Coerce mixed types to string for categorical columns
    - Returns cleaned DataFrame and categorical column list
    """

    def __init__(self, categorical_threshold: int = 50):
        self.cat_threshold = categorical_threshold

    def clean(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
        df = df.copy()
        df = df.drop_duplicates().reset_index(drop=True)

        # unify object-like columns
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype(str)

        # handle missing
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=['object']).columns.tolist()

        for c in num_cols:
            df[c] = df[c].fillna(df[c].median())

        for c in cat_cols:
            df[c] = df[c].fillna(df[c].mode().iloc[0] if not df[c].mode().empty else "Unknown")

        # compress extremely high-cardinality object columns as strings (left as-is)
        return df, cat_cols

    def label_encode(self, X_train, X_test, cat_cols):
        """
        Fit Label/Ordinal encoding on training categorical columns and transform train/test.
        Returns fitted encoders dict for future use.
        """
        encoders = {}
        for c in cat_cols:
            enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            # OrdinalEncoder needs 2D
            X_train[[c]] = enc.fit_transform(X_train[[c]])
            X_test[[c]] = enc.transform(X_test[[c]])
            encoders[c] = enc
        return X_train, X_test, encoders
