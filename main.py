import pandas as pd


class DataPreprocessor:
    def __init__(self, Dframe: "pd.DataFrame") -> None:

        if not isinstance(Dframe, pd.DataFrame):
            raise TypeError(f"Ожидается pandas.DataFrame, получен {type(Dframe)}")

        self.DataFrame = Dframe

    def remove_missing(self, threshold: float = 0.5):
        df = self.DataFrame.copy()
        miss = df.isna().mean()
        df = df.loc[:, miss <= threshold]

        for col in df.columns:
            if df[col].isna().any():
                if df[col].dtype.kind in "biufc":
                    df[col].fillna(df[col].median(), inplace=True)
                else:
                    df[col].fillna(df[col].mode()[0], inplace=True)

        self.DataFrame = df
        return df

    def encode_categorical(self):
        df = self.DataFrame.copy()
        cat_col = df.select_dtypes(include=("object", "category")).columns

        df = pd.get_dummies(df, columns=cat_col)

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


data_dicts = pd.DataFrame({"A": [1, 2, None], "B": [None, 5, 6], "C": ["x", None, "y"]})

df_productss = "i"
p = DataPreprocessor(df_products)

print(p.remove_missing())
