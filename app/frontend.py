import streamlit as st
import pandas as pd
import joblib
import json

# Carregar modelo e dados auxiliares
modelo = joblib.load("model/modelo_regressao.pkl")
with open("model/dropdown_values.json", "r", encoding="utf-8") as f:
    valores_dropdown = json.load(f)
colunas_modelo = pd.read_csv("model/model_columns.csv").squeeze().tolist()

# Mapeamento ordinal atualizado
ordem_tipologias = {'T0': 0, 'T1': 1, 'T2': 2, 'T3': 3, 'T4': 4, 'T5+': 5}

# Interface Streamlit
st.set_page_config(page_title="Previsão Airbnb Lisboa", layout="centered")
st.title("🧠 Previsão de Preço Diário - Airbnb Lisboa")
st.markdown("Simule o preço diário estimado para um novo alojamento local (Airbnb) em Lisboa.")

# Inputs
freguesia = st.selectbox("Escolhe a freguesia:", valores_dropdown["neighbourhood_cleansed"])
tipologia = st.selectbox("Escolhe a tipologia:", valores_dropdown["tipologia"])

if st.button("Prever preço"):
    # Inicializar linha de input
    df_input = pd.DataFrame([[0]*len(colunas_modelo)], columns=colunas_modelo)

    # Ativar dummy da freguesia
    col_freguesia = f"neighbourhood_cleansed_{freguesia}"
    if col_freguesia in df_input.columns:
        df_input.at[0, col_freguesia] = 1
    else:
        st.error(f"❌ Coluna '{col_freguesia}' não encontrada no modelo.")

    # Assumir superhost = não
    col_superhost = "host_is_superhost_no"
    if col_superhost in df_input.columns:
        df_input.at[0, col_superhost] = 1

    # Atribuir valor ordinal à tipologia
    if "tipologia_ordinal" in df_input.columns:
        df_input.at[0, "tipologia_ordinal"] = ordem_tipologias[tipologia]
    else:
        st.error("❌ Coluna 'tipologia_ordinal' não encontrada no modelo.")

    # Aviso para tipologias menos representadas
    if tipologia == "T5+":
        st.info("ℹ️ Nota: Previsões para tipologias T5+ são baseadas em menor quantidade de dados e podem ter maior margem de erro.")

    # Previsão
    preco_previsto = modelo.predict(df_input)[0]
    preco_previsto = max(0, preco_previsto)

    st.success(f"💶 Preço previsto: **{preco_previsto:.2f} €** por noite")
