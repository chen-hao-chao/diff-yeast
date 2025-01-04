import streamlit as st
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import io
import os
from utils import add_outline, pseudo_segmentation
import numpy as np
import imageio.v3 as iio

# Function to adjust color channels of a GIF and optionally add an outline
def adjust_gif_colors_and_outline(gif_path, brightness, red_intensity, green_intensity, blue_intensity, add_outline_):
    gif = Image.open(gif_path)
    frames = []
    new_gif = []

    # Process each frame
    for frame in range(0, gif.n_frames):
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

        # Add outline if checkbox is checked
        adjusted_frame_np = np.array(adjusted_frame)
        print(adjusted_frame_np.shape)
        img = adjusted_frame_np[:,:,0]+adjusted_frame_np[:,:,1]+adjusted_frame_np[:,:,2] // 3
        new_gif.append(np.clip((img - (np.max(img)//30)), a_min=0, a_max=255))
        # if add_outline:
            # adjusted_frame = adjusted_frame.filter(ImageFilter.FIND_EDGES)

        frames.append(adjusted_frame)

    
    if add_outline_:
        masks, _ = pseudo_segmentation(new_gif)
        new_gif = []
        for i in range(0, gif.n_frames):
            new_frame = add_outline(np.array(frames[i]), masks[i], channels=3)
            new_gif.append(Image.fromarray(np.uint8(new_frame)))
        byte_io = io.BytesIO()
        new_gif[0].save(byte_io, format="GIF", save_all=True, append_images=new_gif[1:], loop=0)
        byte_io.seek(0)
    else:
        byte_io = io.BytesIO()
        frames[0].save(byte_io, format="GIF", save_all=True, append_images=frames[1:], loop=0)
        byte_io.seek(0)
    
    return byte_io

# Root directory for GIFs
ROOT_DIR = './test_fig/' #'/scratch/ssd004/scratch/chchao/data/gt_videos/' #

# Collect all GIF files in the directory and its subdirectories
gif_files = []
for root, _, files in os.walk(ROOT_DIR):
    for file in files:
        if file.endswith('.gif'):
            gif_files.append(os.path.join(root, file))

gif_options = {os.path.basename(gif): gif for gif in gif_files}

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

# Checkbox for adding outline
add_outline_ = st.sidebar.checkbox("Add Outline")

# Display selected GIF
if selected_gif_name:
    gif_path = gif_options[selected_gif_name]
    adjusted_gif = adjust_gif_colors_and_outline(gif_path, brightness, red_intensity, green_intensity, blue_intensity, add_outline_)

    # Button to stop GIF playing
    stop_gif = st.sidebar.button("Stop GIF")

    # Display GIF or a placeholder if stopped
    if stop_gif:
        st.image([], caption="GIF Stopped", use_column_width=True)
    else:
        st.image(adjusted_gif, caption=f"Adjusted {selected_gif_name}", use_column_width=True)
