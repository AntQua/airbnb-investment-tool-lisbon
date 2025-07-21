import streamlit as st
import pandas as pd
import joblib
import json

from risk_simulation import simular_risco

# Inicializar variáveis de estado
if "previsao_feita" not in st.session_state:
    st.session_state.previsao_feita = False
if "preco_previsto" not in st.session_state:
    st.session_state.preco_previsto = None

# Carregar dados
df = pd.read_csv("data/listings_lisboa_final.csv")

# Carregar modelo e metadados
modelo = joblib.load("model/modelo.pkl")
with open("model/dropdown_values.json", "r", encoding="utf-8") as f:
    valores_dropdown = json.load(f)
colunas_modelo = pd.read_csv("model/model_columns.csv").squeeze().tolist()

# Mapeamento tipologias
ordem_tipologias = {'T0': 0, 'T1': 1, 'T2': 2, 'T3': 3, 'T4': 4, 'T5+': 5}

# Interface
st.set_page_config(page_title="Previsão Airbnb Lisboa", layout="centered")
st.title("🧠 Previsão de Preço Diário - Airbnb Lisboa")
st.markdown("Simule o preço diário estimado para um novo alojamento local (Airbnb) em Lisboa.")

# Inputs principais
freguesia = st.selectbox("Escolhe a freguesia:", valores_dropdown["neighbourhood_cleansed"])
tipologia = st.selectbox("Escolhe a tipologia:", valores_dropdown["tipologia"])

if st.button("Prever preço"):
    df_input = pd.DataFrame([[0]*len(colunas_modelo)], columns=colunas_modelo)

    col_freguesia = f"neighbourhood_cleansed_{freguesia}"
    if col_freguesia in df_input.columns:
        df_input.at[0, col_freguesia] = 1
    else:
        st.error(f"❌ Coluna '{col_freguesia}' não encontrada no modelo.")

    col_superhost = "host_is_superhost_no"
    if col_superhost in df_input.columns:
        df_input.at[0, col_superhost] = 1

    if "tipologia_ordinal" in df_input.columns:
        df_input.at[0, "tipologia_ordinal"] = ordem_tipologias[tipologia]
    else:
        st.error("❌ Coluna 'tipologia_ordinal' não encontrada no modelo.")

    if tipologia == "T5+":
        st.info("ℹ️ Nota: Previsões para tipologias T5+ são baseadas em menor quantidade de dados e podem ter maior margem de erro.")

    preco_previsto = modelo.predict(df_input)[0]
    preco_previsto = max(0, preco_previsto)
    st.session_state.preco_previsto = preco_previsto
    st.session_state.previsao_feita = True

if st.session_state.previsao_feita and st.session_state.preco_previsto is not None:
    st.success(f"💶 Preço previsto: **{st.session_state.preco_previsto:.2f} €** por noite")

    st.markdown("Deseja realizar uma Simulação e Análise de Risco com base neste preço previsto?")
    simular = st.radio("Seleciona uma opção:", ["Não", "Sim"], index=0, key="sim_risco")

    if simular == "Sim":
        custo_input = st.number_input("💡 Introduz os custos médios por noite (€)", min_value=0.0, step=1.0, key="custo_diario")
        pct_preco = st.number_input("📈 Variação percentual no preço estimado (%)", min_value=-100.0, max_value=100.0, step=1.0, value=0.0)
        pct_ocupacao = st.number_input("📉 Variação percentual na ocupação anual (%)", min_value=-100.0, max_value=100.0, step=1.0, value=0.0)

        if st.button("📊 Simular Risco"):
            try:
                resultado = simular_risco(
                    df, freguesia, tipologia,
                    float(st.session_state.preco_previsto),
                    float(custo_input),
                    delta_price_pct=float(pct_preco),
                    delta_occupancy_pct=float(pct_ocupacao)
                )

                if resultado is None:
                    st.warning("⚠️ Dados insuficientes para simular o risco com esta combinação.")
                else:
                    lucro_medio = resultado['mean_profit']
                    lucro_texto = "lucro" if lucro_medio >= 0 else "prejuízo"
                    direcao_texto = "positivo" if lucro_medio >= 0 else "negativo"

                    st.markdown("## 🔍 Análise de Risco com Simulação")

                    st.markdown(f"""
                    ✅ Um {tipologia} em {freguesia} tem um {lucro_texto} anual esperado **{direcao_texto}**.

                    📅 A simulação foi realizada com base numa ocupação média de **{resultado['occupancy_mean']:.0f} dias/ano** e preço de **{resultado['adjusted_price']:.2f} €** por noite.

                    📉 Probabilidade de prejuízo: **{resultado['prob_loss'] * 100:.1f}%**

                    💰 Com 75% de probabilidade, o {lucro_texto} anual situa-se entre **{resultado['percentiles'][1]:.0f} €** e **{resultado['percentiles'][3]:.0f} €**.

                    ⚠️ O {lucro_texto} pode variar significativamente — desvio padrão: **{resultado['std_profit']:.0f} €**
                    """)

            except Exception as e:
                st.error(f"❌ Erro ao executar simulação: {e}")
