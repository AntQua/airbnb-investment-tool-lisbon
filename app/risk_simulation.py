import numpy as np
import pandas as pd

def simular_risco(df, freguesia, tipologia, predicted_price, cost_per_night, n_sim=1000):
    # Filtrar imóveis semelhantes
    df_filtrado = df[
        (df["neighbourhood_cleansed"] == freguesia) &
        (df["tipologia"] == tipologia)
    ]

    if len(df_filtrado) < 10:
        return None  # Dados insuficientes

    # Estimar parâmetros da distribuição da ocupação
    occupancy_mean = df_filtrado["estimated_occupancy_l365d"].mean()
    occupancy_std = df_filtrado["estimated_occupancy_l365d"].std()

    # Simular ocupação
    np.random.seed(42)
    occupancy = np.random.normal(loc=occupancy_mean, scale=occupancy_std, size=n_sim)
    occupancy = np.clip(occupancy, 0, 365)

    # Cálculo de lucro
    revenue = predicted_price * occupancy
    costs = cost_per_night * occupancy
    profit = revenue - costs

    # Estatísticas
    percentis = np.percentile(profit, [5, 25, 50, 75, 95])
    return {
        "mean_profit": np.mean(profit),
        "std_profit": np.std(profit),
        "prob_loss": float(np.mean(profit < 0)),
        "percentiles": percentis
    }
