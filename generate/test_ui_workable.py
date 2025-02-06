import streamlit as st
from PIL import Image, ImageEnhance, ImageOps
import io
import os

# Function to adjust color channels of a GIF
def adjust_gif_colors(gif_path, brightness, red_intensity, green_intensity, blue_intensity, stop_frame=None):
    gif = Image.open(gif_path)
    frames = []

    # Process each frame
    for frame in range(0, gif.n_frames):
        if stop_frame is not None and frame > stop_frame:
            break
        elif stop_frame is not None and frame < stop_frame:
            continue

        gif.seek(frame)
        frame_image = gif.convert("RGB")

        # Adjust brightness
        enhancer = ImageEnhance.Brightness(frame_image)
        bright_frame = enhancer.enhance(brightness)

        # Adjust color channels
        r, g, b = bright_frame.split()
        r = r.point(lambda i: i * red_intensity)
        g = g.point(lambda i: i * green_intensity)
        b = b.point(lambda i: i * blue_intensity)
        adjusted_frame = Image.merge("RGB", (r, g, b))

        frames.append(adjusted_frame)
        if stop_frame is not None and frame == stop_frame:
            frames.append(adjusted_frame)

    # Save adjusted GIF into a byte stream
    byte_io = io.BytesIO()
    frames[0].save(byte_io, format="GIF", save_all=True, append_images=frames[1:], loop=0)
    byte_io.seek(0)
    return byte_io

import requests
def get_gene_description(gene_name):
    """
    Fetches the text description of a given yeast gene from the SGD API.

    :param gene_name: The standard yeast gene name (e.g., 'CDC28')
    :return: The gene description or an error message.
    """
    base_url = f"https://www.yeastgenome.org/backend/locus/{gene_name}"
    response = requests.get(base_url)
    if response.status_code == 200:
        data = response.json()
        description = data.get("description", "No description available")
        return description
    else:
        return f"Error: Unable to fetch data for {gene_name}. Status code: {response.status_code}"

# Root directories for GIFs
ROOT_DIR_MAIN = './ori_fig'
ROOT_DIR_ALT = './ori_fig_outlined'
ROOT_DIR_MAIN_REVERSE = './test_fig'
ROOT_DIR_ALT_REVERSE = './test_fig_outlined'

# Collect all GIF files in both directories
def get_gif_files(root_dir):
    gif_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.gif'):
                gif_files.append(os.path.join(root, file))
    return gif_files

main_gif_files = get_gif_files(ROOT_DIR_MAIN)
alt_gif_files = get_gif_files(ROOT_DIR_ALT)
main_reverse_gif_files = get_gif_files(ROOT_DIR_MAIN_REVERSE)
alt_reverse_gif_files = get_gif_files(ROOT_DIR_ALT_REVERSE)

# Toggle folder selection
use_alt_folder = st.sidebar.checkbox("Add Outlines")
use_reverse_folder = st.sidebar.checkbox("Reverse")

if use_reverse_folder:
    gif_files = alt_reverse_gif_files if use_alt_folder else main_reverse_gif_files
else:
    gif_files = alt_gif_files if use_alt_folder else main_gif_files
gif_options = {gif.split('/')[-2] + ' -  Variant: ' + str(int(gif.split('/')[-1].split('.')[0].split('_')[0])+1): gif for gif in gif_files}

# Checkbox to stop at a specific frame
stop_at_frame = st.sidebar.checkbox("Stop at a specific frame")
stop_frame = st.sidebar.number_input("Frame to stop at (0-192)", min_value=0, max_value=192, step=1) if stop_at_frame else None


# Sidebar for selecting a GIF
st.sidebar.header("Select a GIF")
selected_gif_name = st.sidebar.radio("Choose a GIF", list(gif_options.keys()))

# Sliders for adjusting parameters
st.sidebar.header("Adjust Parameters")
brightness = 1 #st.sidebar.slider("Brightness", 0.1, 3.0, 1.0, 0.1)
red_intensity = st.sidebar.slider("Red Intensity (Nucleus)", 0.0, 3.0, 1.0, 0.1)
green_intensity = st.sidebar.slider("Green Intensity (Structure)", 0.0, 3.0, 1.0, 0.1)
blue_intensity = 1 #st.sidebar.slider("Blue Intensity", 0.0, 3.0, 1.0, 0.1)

# Display selected GIF
if selected_gif_name:
    # Streamlit app title
    st.title(selected_gif_name)
    gif_path = gif_options[selected_gif_name]
    gene_name = selected_gif_name.split(' ')[0]
    description = get_gene_description(gene_name)
    # print(f"Description of {gene_name}: {description}")

    # Display description
    st.write(f"**Description:** {description}")
    adjusted_gif = adjust_gif_colors(gif_path, brightness, red_intensity, green_intensity, blue_intensity, stop_frame=stop_frame)

    # Display GIF
    st.image(adjusted_gif, caption=f"{selected_gif_name}", use_container_width=True)