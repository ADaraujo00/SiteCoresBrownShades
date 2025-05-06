import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
from collections import Counter
from sklearn.cluster import KMeans
from scipy.spatial import distance
import plotly.express as px
import base64
import os

# Função para verificar se uma cor é próxima de cinza ou branco
def is_gray_or_white(color, threshold=30):
    r, g, b = color
    if abs(r - 255) < threshold and abs(g - 255) < threshold and abs(b - 255) < threshold:
        return True
    if abs(r - g) < threshold and abs(g - b) < threshold and abs(r - b) < threshold:
        return True
    return False

# Função para processar a imagem e calcular as cores normativas
def process_image(image):
    image = image.convert('RGB')
    image = image.resize((image.width // 4, image.height // 4))
    colors = np.array(image.getdata())
    filtered_colors = np.array([color for color in colors if not is_gray_or_white(color)])

    n_colors = 14
    kmeans = KMeans(n_clusters=n_colors, random_state=0, n_init=10, max_iter=300)
    kmeans.fit(filtered_colors)

    quantized_colors = kmeans.cluster_centers_.astype(int)
    labels = kmeans.labels_
    color_count = Counter(labels)
    total_pixels = sum(color_count.values())

    color_df = pd.DataFrame({
        'Color': [tuple(color) for color in quantized_colors],
        'Count': [color_count[i] for i in range(n_colors)]
    })

    color_df['Percentage'] = (color_df['Count'] / total_pixels) * 100

    normative_colors = [
        [77, 62, 59], [93, 71, 63], [108, 81, 67], [124, 91, 71], [140, 102, 76],
        [157, 112, 80], [173, 123, 84], [190, 134, 88], [200, 148, 102], [210, 162, 115],
        [219, 176, 129], [229, 190, 143], [238, 205, 157], [247, 219, 172]
    ]

    def find_closest_color(color, normative_colors):
        closest_color = None
        min_dist = float('inf')
        for norm_color in normative_colors:
            dist = distance.euclidean(color, norm_color)
            if dist < min_dist:
                min_dist = dist
                closest_color = norm_color
        return tuple(closest_color)

    color_df['Closest Normative Color'] = color_df['Color'].apply(
        lambda x: find_closest_color(x, normative_colors))

    normative_color_df = color_df.groupby('Closest Normative Color').agg({
        'Percentage': 'sum'}).reset_index()
    normative_color_df['Closest Normative Color'] = normative_color_df['Closest Normative Color'].apply(str)

    normative_color_df['Color Sort Key'] = normative_color_df['Closest Normative Color'].apply(
        lambda x: eval(x))
    
    # Ordenar o DataFrame pela tonalidade das cores (Color Sort Key)
    normative_color_df.sort_values(by='Color Sort Key', inplace=True)

    # Mapeamento das cores para números (invertido)
    color_to_number = {str(tuple(color)): i for i, color in enumerate(normative_colors[::-1], start=4)}

    normative_color_df['Color Number'] = normative_color_df['Closest Normative Color'].apply(
        lambda x: color_to_number[x])

    # Remove rows with percentage less than 0.5% and 0%
    normative_color_df = normative_color_df[normative_color_df['Percentage'] > 0.5]

    return image, normative_color_df.drop(columns=['Color Sort Key'])

# Função para carregar a imagem da paleta
def load_palette_image():
    with open('paleta.png', 'rb') as f:
        encoded_image = base64.b64encode(f.read()).decode()
    return f'data:image/png;base64,{encoded_image}'

# Interface do Streamlit
st.title("Análise de Cores em Imagens")

uploaded_file = st.file_uploader("Escolha uma imagem...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        image_processed, results_df = process_image(image)

        # Filtrar o DataFrame para remover linhas com porcentagem menor que 0%
        results_df = results_df[results_df['Percentage'] > 0]

        # Ordenar o DataFrame pela tonalidade das cores
        results_df = results_df.sort_values(by='Color Number')

        # Mapeamento das cores para RGB
        color_map = {str(tuple(color)): f'rgb{tuple(color)}' for color in results_df['Closest Normative Color'].apply(eval)}

        # Criar o gráfico de pizza com Plotly
        fig = px.pie(
            results_df,
            names='Closest Normative Color',
            values='Percentage',
            title='Cores Normativas na Imagem por Porcentagem',
            color='Closest Normative Color',
            color_discrete_map=color_map,
            hole=0.3,
            labels={'Closest Normative Color': 'Cor Normativa', 'Percentage': 'Porcentagem (%)'},
            height=800
        )
        fig.update_traces(sort=False)

        # Atualizar o layout do gráfico para mudar o tamanho das fontes
        fig.update_layout(
            title_font_size=24,
            font=dict(
                family="Arial, sans-serif",
                size=25,
                color="black"
            ),
            legend=dict(
                font=dict(
                    size=12
                )
            )
        )

        # Adicionar a imagem da paleta no canto superior direito do gráfico
        fig.add_layout_image(
            dict(
                source=load_palette_image(),
                xref="paper", yref="paper",
                x=1.22, y=0.15,
                sizex=0.30, sizey=0.30,
                xanchor="right", yanchor="top"
            )
        )

        st.image(image, caption='Imagem Carregada', use_column_width=True)
        st.plotly_chart(fig)
        st.dataframe(results_df.round(2))
    except Exception as e:
        st.error(f"Erro ao processar o arquivo: {e}")

# Função para verificar arquivos que começam com 'RIFF'
def check_riff_files():
    riff_files = []
    for i, filename in enumerate(os.listdir('./')):
        if os.path.isfile(filename):
            with open(filename, 'rb') as imageFile:
                if imageFile.read().startswith(b'RIFF'):
                    riff_files.append(filename)
    return riff_files

# Verificar e exibir arquivos que começam com 'RIFF'
riff_files = check_riff_files()
if riff_files:
    st.write("Arquivos que começam com 'RIFF':")
    for i, filename in enumerate(riff_files):
        st.write(f"{i}: {filename} - found!")
else:
    st.write("Nenhum arquivo que começa com 'RIFF' foi encontrado.")
