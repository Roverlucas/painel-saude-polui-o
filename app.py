import streamlit as st
import pandas as pd
import plotly.express as px
import os
import numpy as np
import seaborn as sns
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.stats import pearsonr, spearmanr

# Título do app
st.set_page_config(page_title="Painel de Saúde e Poluição", layout="wide")
st.title("Painel Interativo: Saúde, Poluição e Indicadores Socioeconômicos")

# Carregar dados
@st.cache_data
def carregar_dados():
    df_curitiba = pd.read_csv("Base_Reduzida_Curitiba_PBI.csv")
    df_pg = pd.read_csv("Base_Reduzida_PBI_PontaGrossa.csv")
    df_medianeira = pd.read_csv("Base_Reduzida_PBI_Medianeira.csv")
    df_foz = pd.read_csv("Base_Reduzida_PBI_Foz.csv")
    df_se = pd.read_csv("Indicadores_Socioeconomicos_Cidades.csv")

    for df_cidade in [df_curitiba, df_pg, df_medianeira, df_foz]:
        if "DT_INTER" not in df_cidade.columns and "DATA_ENTRADA" in df_cidade.columns:
            df_cidade.rename(columns={"DATA_ENTRADA": "DT_INTER"}, inplace=True)

    for df_cidade in [df_curitiba, df_pg, df_medianeira, df_foz]:
        if "CLUSTER" not in df_cidade.columns:
            variaveis = ["PM2_5", "INTERNACOES", "OBITOS", "CUSTO_MEDIO", "DURACAO_MEDIA", "UMIDADE", "TEMP_MEDIA"]
            df_tmp = df_cidade.dropna(subset=variaveis).copy()
            df_norm = StandardScaler().fit_transform(df_tmp[variaveis])
            kmeans = KMeans(n_clusters=3, random_state=42).fit(df_norm)
            df_cidade.loc[df_tmp.index, "CLUSTER"] = kmeans.labels_

    df_macro = pd.concat([df_curitiba, df_pg, df_medianeira, df_foz], ignore_index=True)

    if "DT_INTER" not in df_macro.columns and "DATA_ENTRADA" in df_macro.columns:
        df_macro.rename(columns={"DATA_ENTRADA": "DT_INTER"}, inplace=True)
    df_macro["DT_INTER"] = pd.to_datetime(df_macro["DT_INTER"], errors="coerce")

    variaveis_macro = ["PM2_5", "INTERNACOES", "OBITOS", "CUSTO_MEDIO", "DURACAO_MEDIA", "UMIDADE", "TEMP_MEDIA"]
    df_macro_validos = df_macro.dropna(subset=variaveis_macro).copy()
    df_norm_macro = StandardScaler().fit_transform(df_macro_validos[variaveis_macro])
    kmeans_macro = KMeans(n_clusters=4, random_state=42).fit(df_norm_macro)
    df_macro.loc[df_macro_validos.index, "CLUSTER_MACRO"] = kmeans_macro.labels_

    df_macro = df_macro.merge(df_se, how="left", left_on="city", right_on="Cidade")

    df_macro["RAZAO_INT_PM"] = df_macro["INTERNACOES"] / df_macro["PM2_5"]
    df_macro["RAZAO_OB_INT"] = df_macro["OBITOS"] / df_macro["INTERNACOES"]
    df_macro["CUSTO_PM"] = df_macro["CUSTO_MEDIO"] / df_macro["PM2_5"]
    df_macro["INT_PER_CAPITA"] = df_macro["INTERNACOES"] / df_macro["Densidade demográfica (hab/km²)"]

    df_norm_ivp = StandardScaler().fit_transform(df_macro[["PM2_5", "INTERNACOES", "IDHM (2010)", "Saneamento básico (%)"]].fillna(0))
    ivp = 0.4*df_norm_ivp[:,0] + 0.3*df_norm_ivp[:,1] - 0.2*df_norm_ivp[:,2] - 0.1*df_norm_ivp[:,3]
    df_macro["IVP"] = ivp

    return df_macro

df = carregar_dados()

st.sidebar.header("Filtros")
cidade_sel = st.sidebar.selectbox("Selecione uma cidade:", sorted(df["city"].unique()), key="selectbox_cidade")
data_min, data_max = df["DT_INTER"].min(), df["DT_INTER"].max()
data_ini, data_fim = st.sidebar.date_input("Período:", [data_min, data_max], min_value=data_min, max_value=data_max, key="date_input_periodo")
modo_avancado = st.sidebar.checkbox("Ativar Modo Avançado", key="modo_avancado")

df_filtrado = df[(df["city"] == cidade_sel) & (df["DT_INTER"] >= pd.to_datetime(data_ini)) & (df["DT_INTER"] <= pd.to_datetime(data_fim))]

st.markdown(f"## Dashboards - {cidade_sel}")

st.subheader("Métricas Principais")
col1, col2, col3, col4 = st.columns(4)
col1.metric("PM2.5 Médio", f"{df_filtrado['PM2_5'].mean():.1f} µg/m³")
col2.metric("Internações Totais", int(df_filtrado["INTERNACOES"].sum()))
col3.metric("IVP Médio", f"{df_filtrado['IVP'].mean():.2f}")
col4.metric("IDHM", f"{df_filtrado['IDHM (2010)'].mean():.2f}")

st.subheader("Radar das Métricas Derivadas")
cols = ["RAZAO_INT_PM", "RAZAO_OB_INT", "CUSTO_PM", "INT_PER_CAPITA", "IVP"]
df_radar = df.groupby("city")[cols].mean().reset_index()
df_melted = df_radar.melt(id_vars="city", var_name="Métrica", value_name="Valor")
fig_radar = px.line_polar(df_melted[df_melted["city"] == cidade_sel], r="Valor", theta="Métrica", line_close=True)
st.plotly_chart(fig_radar, use_container_width=True)

st.subheader("Correlação IVP x Renda")
fig_scatter = px.scatter(df, x="Renda per capita (R$ mensais)", y="IVP", color="city", trendline="ols")
st.plotly_chart(fig_scatter, use_container_width=True)

st.subheader("Heatmap de Correlação")
df_corr = df_filtrado[cols + ["Renda per capita (R$ mensais)", "IDHM (2010)", "Saneamento básico (%)"]].corr()
fig_heat = px.imshow(df_corr, text_auto=True, color_continuous_scale="RdBu_r")
st.plotly_chart(fig_heat, use_container_width=True)

st.markdown("## Comparativos entre Cidades")
metric_titles = ["Razão Internações / PM2.5", "Razão Óbitos / Internações", "Custo Médio / PM2.5", "Internações per Capita", "Índice de Vulnerabilidade à Poluição"]
for col, title in zip(cols, metric_titles):
    st.subheader(title)
    fig = px.box(df, x="city", y=col, color="city", points="all")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("## Ranking por Cidade")
df_ranking = df.groupby("city")[cols].mean().reset_index().sort_values("IVP", ascending=False)
st.dataframe(df_ranking.round(2))

if modo_avancado:
    st.markdown("---")
    st.markdown("### 🔬 Modo Avançado: Exploração Profunda")

    st.markdown("#### Correlação IVP × Saneamento")
    fig_corr2 = px.scatter(df, x="Saneamento básico (%)", y="IVP", color="city", trendline="ols")
    st.plotly_chart(fig_corr2, use_container_width=True)

    st.markdown("#### Correlação IVP × IDHM")
    fig_corr4 = px.scatter(df, x="IDHM (2010)", y="IVP", color="city", trendline="ols")
    st.plotly_chart(fig_corr4, use_container_width=True)

    st.markdown("#### Correlação Internações × Renda")
    fig_corr5 = px.scatter(df, x="Renda per capita (R$ mensais)", y="INTERNACOES", color="city", trendline="ols")
    st.plotly_chart(fig_corr5, use_container_width=True)

    st.markdown("#### Distribuição de IVP por Cluster Macro")
    fig_violin = px.violin(df, x="CLUSTER_MACRO", y="IVP", color="city", box=True, points="all")
    st.plotly_chart(fig_violin, use_container_width=True)

    st.markdown("#### Distribuição de Renda per Capita por Cidade")
    fig_hist = px.histogram(df, x="Renda per capita (R$ mensais)", color="city", nbins=30, marginal="box")
    st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("#### Distribuição de IDHM por Cidade")
    fig_hist_idhm = px.histogram(df, x="IDHM (2010)", color="city", nbins=20, marginal="box")
    st.plotly_chart(fig_hist_idhm, use_container_width=True)

    st.markdown("#### Distribuição de IVP por Cidade")
    fig_hist_ivp = px.histogram(df, x="IVP", color="city", nbins=20, marginal="box")
    st.plotly_chart(fig_hist_ivp, use_container_width=True)

