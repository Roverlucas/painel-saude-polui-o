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
        if "CLUSTER" not in df_cidade.columns:
            variaveis = ["PM2_5", "INTERNACOES", "OBITOS", "CUSTO_MEDIO", "DURACAO_MEDIA", "UMIDADE", "TEMP_MEDIA"]
            df_tmp = df_cidade.dropna(subset=variaveis).copy()
            df_norm = StandardScaler().fit_transform(df_tmp[variaveis])
            kmeans = KMeans(n_clusters=3, random_state=42).fit(df_norm)
            df_cidade.loc[df_tmp.index, "CLUSTER"] = kmeans.labels_

    df_macro = pd.concat([df_curitiba, df_pg, df_medianeira, df_foz], ignore_index=True)

    # Cluster macro
    variaveis_macro = ["PM2_5", "INTERNACOES", "OBITOS", "CUSTO_MEDIO", "DURACAO_MEDIA", "UMIDADE", "TEMP_MEDIA"]
    df_macro_validos = df_macro.dropna(subset=variaveis_macro).copy()
    df_norm_macro = StandardScaler().fit_transform(df_macro_validos[variaveis_macro])
    kmeans_macro = KMeans(n_clusters=4, random_state=42).fit(df_norm_macro)
    df_macro.loc[df_macro_validos.index, "CLUSTER_MACRO"] = kmeans_macro.labels_

    # Merge socioeconômico
    df_macro = df_macro.merge(df_se, how="left", left_on="city", right_on="Cidade")

    # Métricas derivadas
    df_macro["RAZAO_INT_PM"] = df_macro["INTERNACOES"] / df_macro["PM2_5"]
    df_macro["RAZAO_OB_INT"] = df_macro["OBITOS"] / df_macro["INTERNACOES"]
    df_macro["CUSTO_PM"] = df_macro["CUSTO_MEDIO"] / df_macro["PM2_5"]
    df_macro["INT_PER_CAPITA"] = df_macro["INTERNACOES"] / df_macro["Densidade demográfica (hab/km²)"]

    # IVP (Índice de Vulnerabilidade à Poluição)
    df_norm_ivp = StandardScaler().fit_transform(df_macro[["PM2_5", "INTERNACOES", "IDHM (2010)", "Saneamento básico (%)"]].fillna(0))
    ivp = 0.4*df_norm_ivp[:,0] + 0.3*df_norm_ivp[:,1] - 0.2*df_norm_ivp[:,2] - 0.1*df_norm_ivp[:,3]
    df_macro["IVP"] = ivp

    return df_macro

df = carregar_dados()

# Criação de dashboards por cidade
st.markdown("## Dashboards por Cidade")
cidade_sel = st.selectbox("Selecione uma cidade para visualização detalhada:", sorted(df["city"].unique()))
df_cidade = df[df["city"] == cidade_sel]

# Métricas principais por cidade
st.subheader(f"Métricas Principais - {cidade_sel}")
col1, col2, col3, col4 = st.columns(4)
col1.metric("PM2.5 Médio", f"{df_cidade['PM2_5'].mean():.1f} µg/m³")
col2.metric("Internações Totais", int(df_cidade["INTERNACOES"].sum()))
col3.metric("IVP Médio", f"{df_cidade['IVP'].mean():.2f}")
col4.metric("IDHM", f"{df_cidade['IDHM (2010)'].mean():.2f}")

# Radar por cidade
st.subheader("Radar das Métricas Derivadas")
cols = ["RAZAO_INT_PM", "RAZAO_OB_INT", "CUSTO_PM", "INT_PER_CAPITA", "IVP"]
df_radar = df.groupby("city")[cols].mean().reset_index()
df_melted = df_radar.melt(id_vars="city", var_name="Métrica", value_name="Valor")
fig_radar = px.line_polar(df_melted[df_melted["city"] == cidade_sel], r="Valor", theta="Métrica", line_close=True, title=f"Radar - {cidade_sel}")
st.plotly_chart(fig_radar, use_container_width=True)

# Correlação IVP vs Renda
st.subheader("Correlação IVP x Renda per Capita")
fig_scatter = px.scatter(df, x="Renda per capita (R$ mensais)", y="IVP", color="city", trendline="ols")
st.plotly_chart(fig_scatter, use_container_width=True)

# Heatmap de correlação das métricas por cidade
st.subheader("Heatmap de Correlação por Cidade")
df_corr = df[df["city"] == cidade_sel][cols + ["Renda per capita (R$ mensais)", "IDHM (2010)", "Saneamento básico (%)"]].corr()
fig_heat = px.imshow(df_corr, text_auto=True, color_continuous_scale="RdBu_r", title=f"Correlação - {cidade_sel}")
st.plotly_chart(fig_heat, use_container_width=True)

# Análises complementares já existentes
st.markdown("## Comparativos de Métricas Derivadas")
cols = ["RAZAO_INT_PM", "RAZAO_OB_INT", "CUSTO_PM", "INT_PER_CAPITA", "IVP"]
metric_titles = ["Razão Internações / PM2.5", "Razão Óbitos / Internações", "Custo Médio / PM2.5", "Internações per Capita", "Índice de Vulnerabilidade à Poluição (IVP)"]
for col, title in zip(cols, metric_titles):
    st.subheader(title)
    fig = px.box(df, x="city", y=col, color="city", points="all", title=title)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("## Ranking por Cidade (Médias das Métricas Derivadas)")
df_ranking = df.groupby("city")[cols].mean().reset_index().sort_values("IVP", ascending=False)
st.dataframe(df_ranking.style.format("{:.2f}"))




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
        if "CLUSTER" not in df_cidade.columns:
            variaveis = ["PM2_5", "INTERNACOES", "OBITOS", "CUSTO_MEDIO", "DURACAO_MEDIA", "UMIDADE", "TEMP_MEDIA"]
            df_tmp = df_cidade.dropna(subset=variaveis).copy()
            df_norm = StandardScaler().fit_transform(df_tmp[variaveis])
            kmeans = KMeans(n_clusters=3, random_state=42).fit(df_norm)
            df_cidade.loc[df_tmp.index, "CLUSTER"] = kmeans.labels_

    df_macro = pd.concat([df_curitiba, df_pg, df_medianeira, df_foz], ignore_index=True)

    # Cluster macro
    variaveis_macro = ["PM2_5", "INTERNACOES", "OBITOS", "CUSTO_MEDIO", "DURACAO_MEDIA", "UMIDADE", "TEMP_MEDIA"]
    df_macro_validos = df_macro.dropna(subset=variaveis_macro).copy()
    df_norm_macro = StandardScaler().fit_transform(df_macro_validos[variaveis_macro])
    kmeans_macro = KMeans(n_clusters=4, random_state=42).fit(df_norm_macro)
    df_macro.loc[df_macro_validos.index, "CLUSTER_MACRO"] = kmeans_macro.labels_

    # Merge socioeconômico
    df_macro = df_macro.merge(df_se, how="left", left_on="city", right_on="Cidade")

    # Métricas derivadas
    df_macro["RAZAO_INT_PM"] = df_macro["INTERNACOES"] / df_macro["PM2_5"]
    df_macro["RAZAO_OB_INT"] = df_macro["OBITOS"] / df_macro["INTERNACOES"]
    df_macro["CUSTO_PM"] = df_macro["CUSTO_MEDIO"] / df_macro["PM2_5"]
    df_macro["INT_PER_CAPITA"] = df_macro["INTERNACOES"] / df_macro["Densidade demográfica (hab/km²)"]

    # IVP (Índice de Vulnerabilidade à Poluição)
    df_norm_ivp = StandardScaler().fit_transform(df_macro[["PM2_5", "INTERNACOES", "IDHM (2010)", "Saneamento básico (%)"]].fillna(0))
    ivp = 0.4*df_norm_ivp[:,0] + 0.3*df_norm_ivp[:,1] - 0.2*df_norm_ivp[:,2] - 0.1*df_norm_ivp[:,3]
    df_macro["IVP"] = ivp

    return df_macro
    
@st.cache_data
def carregar_dados():
    df_curitiba = pd.read_csv("Base_Reduzida_Curitiba_PBI.csv")
    df_pg = pd.read_csv("Base_Reduzida_PBI_PontaGrossa.csv")
    df_medianeira = pd.read_csv("Base_Reduzida_PBI_Medianeira.csv")
    df_foz = pd.read_csv("Base_Reduzida_PBI_Foz.csv")
    df_se = pd.read_csv("Indicadores_Socioeconomicos_Cidades.csv")

    for df_cidade in [df_curitiba, df_pg, df_medianeira, df_foz]:
        if "CLUSTER" not in df_cidade.columns:
            variaveis = ["PM2_5", "INTERNACOES", "OBITOS", "CUSTO_MEDIO", "DURACAO_MEDIA", "UMIDADE", "TEMP_MEDIA"]
            df_tmp = df_cidade.dropna(subset=variaveis).copy()
            df_norm = StandardScaler().fit_transform(df_tmp[variaveis])
            kmeans = KMeans(n_clusters=3, random_state=42).fit(df_norm)
            df_cidade.loc[df_tmp.index, "CLUSTER"] = kmeans.labels_

    df_macro = pd.concat([df_curitiba, df_pg, df_medianeira, df_foz], ignore_index=True)

    # Cluster macro
    variaveis_macro = ["PM2_5", "INTERNACOES", "OBITOS", "CUSTO_MEDIO", "DURACAO_MEDIA", "UMIDADE", "TEMP_MEDIA"]
    df_macro_validos = df_macro.dropna(subset=variaveis_macro).copy()
    df_norm_macro = StandardScaler().fit_transform(df_macro_validos[variaveis_macro])
    kmeans_macro = KMeans(n_clusters=4, random_state=42).fit(df_norm_macro)
    df_macro.loc[df_macro_validos.index, "CLUSTER_MACRO"] = kmeans_macro.labels_

    # Merge socioeconômico
    df_macro = df_macro.merge(df_se, how="left", left_on="city", right_on="Cidade")

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

# Tipo de cluster
cluster_tipo = st.radio("Tipo de clusterização:", ["Cluster por Cidade", "Cluster Macro"])
cluster_col = "CLUSTER" if cluster_tipo == "Cluster por Cidade" else "CLUSTER_MACRO"
clusters = df_filtro[cluster_col].dropna().unique()
cluster_opcoes = ["Todos"] + sorted([str(c) for c in clusters if pd.notnull(c)])
cluster_sel = st.selectbox("Selecione o cluster:", options=cluster_opcoes)
if cluster_sel != "Todos":
    df_filtro = df_filtro[df_filtro[cluster_col] == int(cluster_sel)]

# KPIs
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total de Internações", int(df_filtro["INTERNACOES"].sum()))
col2.metric("Total de Óbitos", int(df_filtro["OBITOS"].sum()))
col3.metric("PM2.5 Médio", f"{df_filtro['PM2_5'].mean():.1f} µg/m³")
col4.metric("Umidade Média", f"{df_filtro['UMIDADE'].mean():.1f}%")

col5, col6, col7, col8 = st.columns(4)
col5.metric("Custo Médio", f"R$ {df_filtro['CUSTO_MEDIO'].mean():,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
col6.metric("Duração Média", f"{df_filtro['DURACAO_MEDIA'].mean():.1f} dias")
col7.metric("Velocidade Vento Média", f"{df_filtro['VEL_MEDIA'].mean():.1f} m/s")
col8.metric("Temperatura Média", f"{df_filtro['TEMP_MEDIA'].mean():.1f} °C")

# KPIs Socioeconômicos
st.markdown("### Indicadores Socioeconômicos da Cidade")
if cidade_sel != "Todas":
    df_se_city = df[df["city"] == cidade_sel].iloc[0]
    col9, col10, col11 = st.columns(3)
    col9.metric("IDHM", df_se_city["IDHM (2010)"])
    col10.metric("Renda Per Capita (R$)", f"{df_se_city['Renda per capita (R$ mensais)']:.2f}")
    col11.metric("Taxa de Alfabetização (%)", f"{df_se_city['Taxa de alfabetização (%)']:.1f}%")

st.markdown("---")

# Comparativo entre cidades
st.markdown("### Comparativo entre Cidades - Indicadores Socioeconômicos")
cols_se = ["IDHM (2010)", "Renda per capita (R$ mensais)", "Taxa de alfabetização (%)", "Taxa de urbanização (%)", "Densidade demográfica (hab/km²)", "Saneamento básico (%)"]
df_comp_se = df[["city"] + cols_se].drop_duplicates()
fig_se = px.bar(df_comp_se.melt(id_vars="city"), x="city", y="value", color="variable",
                barmode="group", title="Comparação dos Indicadores Socioeconômicos")
st.plotly_chart(fig_se, use_container_width=True)

# Estatísticas por cluster
df_stats = df.groupby(cluster_col)[["PM2_5", "INTERNACOES", "OBITOS", "CUSTO_MEDIO", "DURACAO_MEDIA", "UMIDADE", "TEMP_MEDIA"] + cols_se].mean().round(2)
st.markdown("### Estatísticas Médias por Cluster")
st.dataframe(df_stats)

# Correlação entre variáveis
st.markdown("### Correlação entre Indicadores de Poluição, Saúde e Socioeconômicos")
variaveis_corr = ["PM2_5", "INTERNACOES", "OBITOS", "CUSTO_MEDIO", "DURACAO_MEDIA", "UMIDADE", "TEMP_MEDIA"] + cols_se
df_corr = df[variaveis_corr].dropna()
corr_matrix = df_corr.corr(method="pearson")
fig_corr = px.imshow(corr_matrix, text_auto=True, title="Matriz de Correlação (Pearson)", aspect="auto")
st.plotly_chart(fig_corr, use_container_width=True)

# Gráficos preservados
fig_series = px.line(df_filtro, x="DATA_ENTRADA", y="INTERNACOES", color="city", title="Série Temporal de Internações")
st.plotly_chart(fig_series, use_container_width=True)

fig_pm = px.line(df_filtro, x="DATA_ENTRADA", y="PM2_5", color="city", title="Série Temporal de PM2.5")
st.plotly_chart(fig_pm, use_container_width=True)

fig_disp = px.scatter(df_filtro, x="PM2_5", y="INTERNACOES", color=cluster_col, title="Dispersão PM2.5 vs Internações")
st.plotly_chart(fig_disp, use_container_width=True)

if "lat" in df_filtro.columns and "long" in df_filtro.columns:
    fig_map = px.scatter_mapbox(df_filtro, lat="lat", lon="long", color=cluster_col,
                                 size="INTERNACOES", zoom=5,
                                 hover_name="city", hover_data=["PM2_5", "INTERNACOES"],
                                 mapbox_style="carto-positron", title="Mapa de Internações e Poluição")
    st.plotly_chart(fig_map, use_container_width=True)

# Mostrar link com ngrok
auth_token = os.environ.get("NGROK_AUTH_TOKEN")
if auth_token:
    from pyngrok import ngrok
    ngrok.set_auth_token(auth_token)
    public_url = ngrok.connect(8501)
    st.success(f"🔗 Painel disponível em: {public_url}")

