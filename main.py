import pandas as pd


class DataPreprocessor:
    def __init__(self, Dframe: "pd.DataFrame") -> None:

        if not isinstance(Dframe, pd.DataFrame):
            raise TypeError(f"Ожидается pandas.DataFrame, получен {type(Dframe)}")
        self.DataFrame = Dframe.copy()
        self.dropped_columns = []
        self.ohe_columns = []

    def remove_missing(self, threshold: float = 0.5):
        if not 0 <= threshold <= 1:
            raise ValueError("threshold должен быть [0:1], а не {threshold}")
        df = self.DataFrame.copy()
        miss = df.isna().mean()
        df = df.loc[:, miss <= threshold]
        self.dropped_columns = miss[miss > threshold].index.tolist()

        for col in df.columns:
            if df[col].isna().any():
                if df[col].dtype.kind in "biufc":
                    df.loc[:, col] = df[col].fillna(df[col].median())
                else:  
                    df.loc[:, col] = df[col].fillna(df[col].mode()[0])

        self.DataFrame = df
        return df

    def encode_categorical(self):
        df = self.DataFrame.copy()
        cat_col = df.select_dtypes(include=("object", "category")).columns

        df = pd.get_dummies(df, columns=cat_col)
        self.ohe_columns = df.columns.tolist()
        self.DataFrame = df
        return df

    def normalize_numeric(self, method: str = "minmax"):
        if method not in ("std", "minmax"):
            raise TypeError(f"Ожидается std/minmax, получен {method}")
        df = self.DataFrame.copy()
        num_cols = df.select_dtypes(include="number").columns

        for col in num_cols:
            if method == "minmax":
                df[col] = (df[col] - df[col].min()) / (
                    df[col].max() - df[col].min()
                )
            elif method == "std":
                df[col] = (df[col] - df[col].mean()) / df[col].std()
        self.DataFrame = df
        return df

    def fit_transform(self, threshold: float = 0.5, method: str = "minmax"):
        try:
            self.remove_missing(threshold)
            self.encode_categorical()
            self.normalize_numeric(method)
            return self.DataFrame
        except Exception as e:
            print("Произошла ошибка при обработке DataFrame:", e)
            return None

