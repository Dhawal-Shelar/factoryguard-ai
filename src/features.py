import pandas as pd


def create_features(df):

    df = df.sort_values(["machine_id", "timestamp"])

    for window in [6, 12]:   # removed 1 here

        df[f"temp_roll_mean_{window}"] = (
            df.groupby("machine_id")["temperature"]
            .transform(lambda x: x.rolling(window=window).mean())
        )

        df[f"vib_roll_mean_{window}"] = (
            df.groupby("machine_id")["vibration"]
            .transform(lambda x: x.rolling(window=window).mean())
        )

        df[f"press_roll_mean_{window}"] = (
            df.groupby("machine_id")["pressure"]
            .transform(lambda x: x.rolling(window=window).mean())
        )

        df[f"temp_roll_std_{window}"] = (
            df.groupby("machine_id")["temperature"]
            .transform(lambda x: x.rolling(window=window).std())
        )

    # EMA
    df["temp_ema_6"] = (
        df.groupby("machine_id")["temperature"]
        .transform(lambda x: x.ewm(span=6, adjust=False).mean())
    )

    # Lag features
    df["temp_lag_1"] = df.groupby("machine_id")["temperature"].shift(1)
    df["temp_lag_2"] = df.groupby("machine_id")["temperature"].shift(2)

    return df

def create_target_label(df):
  
    df = df.sort_values(["machine_id", "timestamp"])

    df["failure_next_24h"] = (
        df.groupby("machine_id")["failure"]
        .transform(lambda x: x.rolling(window=24, min_periods=1).max().shift(-23))
    )

    df["failure_next_24h"] = df["failure_next_24h"].fillna(0)

    return df