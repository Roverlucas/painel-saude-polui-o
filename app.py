import streamlit as st
import pandas as pd
import plotly.express as px
import os
import numpy as np
import seaborn as sns
import plotly.graph_objects as go

# Título do app
st.set_page_config(page_title="Painel de Saúde e Poluição", layout="wide")
st.title("Painel Interativo: Saúde e Poluição")

# Carregar dados
@st.cache_data
def carregar_dados():
    df_curitiba = pd.read_csv("Base_Reduzida_Curitiba_PBI.csv")
    df_pg = pd.read_csv("Base_Reduzida_PBI_PontaGrossa.csv")
    df_medianeira = pd.read_csv("Base_Reduzida_PBI_Medianeira.csv")
    df_foz = pd.read_csv("Base_Reduzida_PBI_Foz.csv")

    df_macro = pd.concat([df_curitiba, df_pg, df_medianeira, df_foz], ignore_index=True)
    df_macro["CLUSTER_MACRO"] = df_macro["CLUSTER"]
    return df_macro

df = carregar_dados()

# Filtros
cidades = df["city"].unique()
cidade_sel = st.selectbox("Selecione a cidade:", options=["Todas"] + sorted(list(cidades)))
df_filtro = df.copy()
if cidade_sel != "Todas":
    df_filtro = df[df["city"] == cidade_sel]

# Filtro por data
st.markdown("### Filtro por período")
data_min = pd.to_datetime(df_filtro["DATA_ENTRADA"].min())
data_max = pd.to_datetime(df_filtro["DATA_ENTRADA"].max())
data_sel = st.date_input("Selecione o intervalo de datas:", [data_min, data_max], min_value=data_min, max_value=data_max)
df_filtro = df_filtro[(pd.to_datetime(df_filtro["DATA_ENTRADA"]) >= pd.to_datetime(data_sel[0])) & (pd.to_datetime(df_filtro["DATA_ENTRADA"]) <= pd.to_datetime(data_sel[1]))]

# Filtro por tipo de cluster
cluster_tipo = st.radio("Tipo de clusterização:", ["Cluster por Cidade", "Cluster Macro"])
cluster_col = "CLUSTER" if cluster_tipo == "Cluster por Cidade" else "CLUSTER_MACRO"
clusters = df_filtro[cluster_col].dropna().unique()
cluster_opcoes = ["Todos"] + sorted([str(c) for c in clusters if pd.notnull(c)])
cluster_sel = st.selectbox("Selecione o cluster:", options=cluster_opcoes)
if cluster_sel != "Todos":
    df_filtro = df_filtro[df_filtro[cluster_col] == cluster_sel]

# Validar colunas ambientais
colunas_amb = ["VEL_MEDIA", "TEMP_MEDIA"]
for col in colunas_amb:
    if col not in df_filtro.columns:
        df_filtro[col] = None

# KPIs de saúde e ambientais
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total de Internações", int(df_filtro["INTERNACOES"].sum()))
col2.metric("Total de Óbitos", int(df_filtro["OBITOS"].sum()))
col3.metric("PM2.5 Médio", f"{df_filtro['PM2_5'].mean():.1f} µg/m³")
col4.metric("Umidade Média", f"{df_filtro['UMIDADE'].mean():.1f}%")

col5, col6, col7, col8 = st.columns(4)
col5.metric("Custo Médio", f"R$ {df_filtro['CUSTO_MEDIO'].mean():,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
col6.metric("Duração Média", f"{df_filtro['DURACAO_MEDIA'].mean():.1f} dias")
col7.metric("Velocidade Vento Média", f"{df_filtro['VEL_MEDIA'].mean():.1f} m/s" if df_filtro['VEL_MEDIA'].notna().any() else "N/A")
col8.metric("Temperatura Média", f"{df_filtro['TEMP_MEDIA'].mean():.1f} °C" if df_filtro['TEMP_MEDIA'].notna().any() else "N/A")

st.markdown("---")

# Comparativo por cluster
st.markdown("### Tabela de características por cluster")
col_cluster = cluster_col
df_stats = df.groupby(col_cluster)[["PM2_5", "UMIDADE", "TEMP_MEDIA", "VEL_MEDIA", "INTERNACOES", "OBITOS", "CUSTO_MEDIO", "DURACAO_MEDIA"]].agg(["mean", "std", "min", "max"]).round(1)
st.dataframe(df_stats)

# Correlação entre variáveis
st.markdown("### Matriz de Correlação")
df_corr = df_filtro[["PM2_5", "INTERNACOES", "OBITOS", "CUSTO_MEDIO", "DURACAO_MEDIA", "UMIDADE", "TEMP_MEDIA"]].corr().round(2)
fig_corr = px.imshow(df_corr, text_auto=True, color_continuous_scale="PuOr", title="Correlação entre Indicadores")
st.plotly_chart(fig_corr, use_container_width=True)

# Índice de risco
st.markdown("### Ranking de Risco por Cluster")
df_risco = df.copy()
df_risco["INDICE_RISCO"] = (
    df_risco["PM2_5"].rank() * 0.4 +
    df_risco["INTERNACOES"].rank() * 0.3 +
    df_risco["OBITOS"].rank() * 0.2 +
    df_risco["CUSTO_MEDIO"].rank() * 0.1
)
df_risco_grouped = df_risco.groupby(col_cluster)["INDICE_RISCO"].mean().reset_index().sort_values("INDICE_RISCO", ascending=False)
st.dataframe(df_risco_grouped.rename(columns={"INDICE_RISCO": "Índice de Risco Médio"}))

# Interpretador textual
if cluster_sel != "Todos":
    st.markdown(f"### Interpretação do Cluster {cluster_sel}")
    with st.expander("Ver interpretação"):
        st.write(f"• Média de PM2.5: {df_filtro['PM2_5'].mean():.2f} — {'alta' if df_filtro['PM2_5'].mean() > 25 else 'moderada'}")
        st.write(f"• Internações: {df_filtro['INTERNACOES'].sum()} casos no período selecionado")
        st.write(f"• Mortalidade: {df_filtro['OBITOS'].sum()} óbitos")
        st.write(f"• Custo médio: R$ {df_filtro['CUSTO_MEDIO'].mean():,.2f}")

# Mostrar link para acesso quando rodar localmente com ngrok
auth_token = os.environ.get("NGROK_AUTH_TOKEN")
if auth_token:
    from pyngrok import ngrok
    ngrok.set_auth_token(auth_token)
    public_url = ngrok.connect(8501)
    st.success(f"🔗 Painel disponível em: {public_url}")

