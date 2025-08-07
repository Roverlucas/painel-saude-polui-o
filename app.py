import streamlit as st
import pandas as pd
import plotly.express as px
import os

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

# Filtro por cluster
clusters = df_filtro["CLUSTER"].dropna().unique()
cluster_sel = st.selectbox("Selecione o cluster:", options=["Todos"] + sorted([str(c) for c in clusters if pd.notnull(c)]))
if cluster_sel != "Todos":
    df_filtro = df_filtro[df_filtro["CLUSTER"] == cluster_sel]

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

# Comparação entre cidades (boxplots)
st.markdown("### Comparativo entre cidades")
col9, col10 = st.columns(2)
with col9:
    fig_comp1 = px.box(df, x="city", y="PM2_5", color="city", title="Distribuição de PM2.5 por Cidade")
    st.plotly_chart(fig_comp1, use_container_width=True)
with col10:
    fig_comp2 = px.box(df, x="city", y="INTERNACOES", color="city", title="Distribuição de Internações por Cidade")
    st.plotly_chart(fig_comp2, use_container_width=True)

# Séries temporais interativas
st.markdown("### Séries Temporais por Cidade")
cidades_multiselect = st.multiselect("Escolha as cidades para comparar:", options=sorted(df["city"].unique()), default=sorted(df["city"].unique()))
df_series = df[df["city"].isin(cidades_multiselect)]
fig_series = px.line(df_series, x="DATA_ENTRADA", y="INTERNACOES", color="city", title="Internações ao longo do tempo por cidade")
st.plotly_chart(fig_series, use_container_width=True)

st.markdown("---")

# Linha do tempo da cidade filtrada
fig_tempo = px.line(df_filtro, x="DATA_ENTRADA", y=["INTERNACOES", "PM2_5"],
                    labels={"value": "Contagem", "variable": "Variável"},
                    title=f"Internações e PM2.5 ao longo do tempo - {cidade_sel if cidade_sel != 'Todas' else 'Todas as cidades'}")
st.plotly_chart(fig_tempo, use_container_width=True)

# Linha inferior: Mapa e Dispersão
col11, col12 = st.columns(2)
with col11:
    fig_map = px.scatter_mapbox(df_filtro,
                                 lat="lat", lon="long", color="CLUSTER",
                                 size="INTERNACOES", zoom=6,
                                 hover_name="city", hover_data=["PM2_5", "INTERNACOES"],
                                 mapbox_style="carto-positron",
                                 title="Mapa de Internações e Poluição")
    st.plotly_chart(fig_map, use_container_width=True)

with col12:
    fig_disp = px.scatter(df_filtro, x="PM2_5", y="INTERNACOES", color="CLUSTER",
                          title="Dispersão PM2.5 vs Internações")
    st.plotly_chart(fig_disp, use_container_width=True)

# Barras por cluster
fig_bar = px.bar(df_filtro.groupby("CLUSTER")["INTERNACOES"].sum().reset_index(),
                 x="CLUSTER", y="INTERNACOES",
                 title="Total de internações por cluster")
st.plotly_chart(fig_bar, use_container_width=True)

# Radar - comparativo de saúde e ambiente por cluster
df_radar = df_filtro.groupby("CLUSTER")[["PM2_5", "UMIDADE", "TEMP_MEDIA", "VEL_MEDIA", "CUSTO_MEDIO", "DURACAO_MEDIA"]].mean().reset_index()
df_radar_melted = df_radar.melt(id_vars="CLUSTER", var_name="Métrica", value_name="Valor")
fig_radar = px.line_polar(df_radar_melted, r="Valor", theta="Métrica", color="CLUSTER",
                          line_close=True, title="Comparativo de Métricas por Cluster")
st.plotly_chart(fig_radar, use_container_width=True)

# Heatmap por cluster e métrica
st.markdown("### Heatmap de indicadores por cluster")
df_heatmap = df_filtro.groupby("CLUSTER")[["PM2_5", "UMIDADE", "TEMP_MEDIA", "VEL_MEDIA", "INTERNACOES", "OBITOS"]].mean().round(1)
fig_heatmap = px.imshow(df_heatmap, text_auto=True, color_continuous_scale="RdBu_r",
                        aspect="auto", title="Matriz de Indicadores Médios por Cluster")
st.plotly_chart(fig_heatmap, use_container_width=True)

# Tabela de características por cluster
st.markdown("### Tabela de características por cluster")
st.dataframe(df_heatmap.reset_index())

# Mostrar link para acesso quando rodar localmente com ngrok
auth_token = os.environ.get("NGROK_AUTH_TOKEN")
if auth_token:
    from pyngrok import ngrok
    ngrok.set_auth_token(auth_token)
    public_url = ngrok.connect(8501)
    st.success(f"🔗 Painel disponível em: {public_url}")
