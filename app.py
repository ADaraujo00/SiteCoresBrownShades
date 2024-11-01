import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
from collections import Counter
from sklearn.cluster import KMeans
from scipy.spatial import distance
import plotly.express as px
import base64
import io

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
    filtered_colors = np.array(
        [color for color in colors if not is_gray_or_white(color)])

    n_colors = 14
    kmeans = KMeans(n_clusters=n_colors, random_state=0,
                    n_init=10, max_iter=300)
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
        [77, 62, 59], [93, 71, 63], [108, 81, 67], [
            124, 91, 71], [140, 102, 76],
        [157, 112, 80], [173, 123, 84], [190, 134, 88], [
            200, 148, 102], [210, 162, 115],
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
    normative_color_df['Closest Normative Color'] = normative_color_df['Closest Normative Color'].apply(
        str)

    normative_color_df['Color Sort Key'] = normative_color_df['Closest Normative Color'].apply(
        lambda x: eval(x))
    normative_color_df.sort_values(by='Color Sort Key', inplace=True)

    # Mapeamento das cores para números (invertido)
    color_to_number = {str(tuple(color)): i for i,
                       color in enumerate(normative_colors[::-1], start=4)}

    normative_color_df['Color Number'] = normative_color_df['Closest Normative Color'].apply(
        lambda x: color_to_number[x])

    # Remove rows with zero percentage
    normative_color_df = normative_color_df[normative_color_df['Percentage'] > 0]

    return image, normative_color_df.drop(columns=['Color Sort Key'])

# Função para carregar a imagem da paleta
def load_palette_image():
    with open('paleta.png', 'rb') as f:
        encoded_image = base64.b64encode(f.read()).decode()
    return f'data:image/png;base64,{encoded_image}'

# Interface do Streamlit
st.title("Análise de Cores em Imagens")

uploaded_files = st.file_uploader("Escolha uma ou mais imagens...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        image_processed, results_df = process_image(image)

        # Desconsiderar cores com porcentagem menor que 1%
        results_df = results_df[results_df['Percentage'] >= 1]

        color_map = {str(tuple(color)): f'rgb{tuple(color)}' for color in results_df['Closest Normative Color'].apply(eval)}

        fig = px.bar(
            results_df,
            x='Percentage',
            y=results_df['Color Number'],
            orientation='h',
            title=f'Normative Colors in Image: {uploaded_file.name}',
            labels={'Percentage': 'Percentage(%)', 'y': 'Color Number'},
            text=results_df['Percentage'].apply(lambda x: f'{x:.2f}%'),
            color=results_df['Closest Normative Color'],
            color_discrete_map=color_map,
            height=800,
            width=1000,
        )

        fig.update_layout(yaxis={'categoryorder': 'array',
                                 'categoryarray': results_df['Color Number'][::-1]},
                          plot_bgcolor='#FFFFFF', paper_bgcolor='#FFFFFF', font=dict(color='black'))

        # Adiciona a imagem da paleta no canto superior direito do gráfico (ajustada)
        fig.add_layout_image(
            dict(
                source=load_palette_image(),
                xref="paper", yref="paper",
                x=1.25, y=0.5,
                sizex=0.30, sizey=0.30,
                xanchor="right", yanchor="top"
            )
        )

        st.image(image, caption=f'Imagem Carregada: {uploaded_file.name}', use_column_width=True)
        st.plotly_chart(fig)
        st.dataframe(results_df.round(2))