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

# Root directories for GIFs
ROOT_DIR_MAIN = './test_fig'
ROOT_DIR_ALT = './test_fig_outlined'

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

# Toggle folder selection
use_alt_folder = st.sidebar.checkbox("Add Outlines")
gif_files = alt_gif_files if use_alt_folder else main_gif_files
gif_options = {gif.split('/')[-2] + '/' + gif.split('/')[-1].split('.')[0]: gif for gif in gif_files}

# Checkbox to stop at a specific frame
stop_at_frame = st.sidebar.checkbox("Stop at a specific frame")
stop_frame = st.sidebar.number_input("Frame to stop at (0-192)", min_value=0, step=1) if stop_at_frame else None


# Streamlit app title
st.title("GIF Parameter Slider")

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
    gif_path = gif_options[selected_gif_name]
    adjusted_gif = adjust_gif_colors(gif_path, brightness, red_intensity, green_intensity, blue_intensity, stop_frame=stop_frame)

    # Display GIF
    st.image(adjusted_gif, caption=f"{selected_gif_name}", use_column_width=True)