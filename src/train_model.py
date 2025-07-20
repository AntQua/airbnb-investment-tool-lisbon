import pandas as pd
import numpy as np
import json
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# Carregar dados
df = pd.read_csv("data/listings_lisboa_final.csv")

# Consolidar T5 e T6 como "T5+"
df['tipologia'] = df['tipologia'].replace({'T5': 'T5+', 'T6': 'T5+'})

# Mapear tipologia como ordinal
ordem_tipologias = {'T0': 0, 'T1': 1, 'T2': 2, 'T3': 3, 'T4': 4, 'T5+': 5}
df['tipologia_ordinal'] = df['tipologia'].map(ordem_tipologias)

# Guardar valores únicos para o frontend
dropdown_values = {
    "neighbourhood_cleansed": sorted(df['neighbourhood_cleansed'].dropna().unique().tolist()),
    "tipologia": sorted(df['tipologia'].dropna().unique().tolist())
}
with open("model/dropdown_values.json", "w", encoding='utf-8') as f:
    json.dump(dropdown_values, f, ensure_ascii=False, indent=2)

# Codificar variáveis categóricas restantes
df_dummies = pd.get_dummies(df, columns=['neighbourhood_cleansed', 'host_is_superhost'], drop_first=False)

# Features numéricas + dummies
features_numericas = [
    'accommodates', 'beds', 'bathrooms', 'number_of_reviews', 'review_scores_rating', 'estimated_occupancy_l365d', 'atratividade_bairro', 'dist_centro_km', 'tipologia_ordinal'
]
X = df_dummies[features_numericas +
               [col for col in df_dummies.columns if col.startswith('neighbourhood_cleansed_') or col.startswith('host_is_superhost_')]]
y = df_dummies['price']

# Dividir treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar modelo
modelo = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
modelo.fit(X_train, y_train)

# Avaliação
# y_pred = modelo.predict(X_test)
# rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# r2 = r2_score(y_test, y_pred)

# Distribuição de tipologia por freguesia
# distribuicao = df.groupby(['neighbourhood_cleansed', 'tipologia']).size().unstack(fill_value=0)

# Visualizar
# print(distribuicao)


# Erro médio por tipologia
# X_test['preco_real'] = y_test
# X_test['preco_previsto'] = y_pred
# X_test['erro'] = X_test['preco_previsto'] - X_test['preco_real']
# print(X_test.groupby('tipologia_ordinal')['erro'].mean())

# Output
print("✅ Modelo treinado com sucesso")
# print(f"R²: {r2:.3f}")
# print(f"RMSE: {rmse:.2f} €")

# Guardar modelo
os.makedirs("model", exist_ok=True)
joblib.dump(modelo, "model/modelo.pkl")
print("📦 Modelo salvo em model/modelo.pkl")

# ✅ Guardar as colunas usadas no modelo
X.columns.to_series().to_csv("model/model_columns.csv", index=False)
print("🗂️ Colunas do modelo guardadas em model/model_columns.csv")
