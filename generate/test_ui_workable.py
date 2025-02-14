import streamlit as st
from PIL import Image, ImageEnhance
import io
import os
import numpy as np
import requests

# Function to process GIF and synchronize Red & Green GIFs
def process_gif(gif_path, brightness, contrast, red_intensity, green_intensity, blue_intensity, 
                red_min_val, red_max_val, green_min_val, green_max_val, stop_frame=None, 
                split_channels=False, frame_duration=100):
    gif = Image.open(gif_path)
    frames = []
    red_frames = []
    green_frames = []
    combined_frames = []

    for frame in range(gif.n_frames):
        if stop_frame is not None and frame > stop_frame:
            break
        elif stop_frame is not None and frame < stop_frame:
            continue

        gif.seek(frame)
        frame_image = gif.convert("RGB")

        # Adjust brightness
        enhancer = ImageEnhance.Brightness(frame_image)
        bright_frame = enhancer.enhance(brightness)

        # Adjust contrast
        contrast_enhancer = ImageEnhance.Contrast(bright_frame)
        contrast_frame = contrast_enhancer.enhance(contrast)

        # Adjust color channels
        r, g, b = contrast_frame.split()
        r = r.point(lambda i: i * red_intensity)
        g = g.point(lambda i: i * green_intensity)
        b = b.point(lambda i: i * blue_intensity)
        adjusted_frame = Image.merge("RGB", (r, g, b))

        # Convert to NumPy for contrast adjustment
        frame_np = np.array(adjusted_frame, dtype=np.float32)

        # ImageJ-style contrast adjustment for Red and Green channels
        def adjust_contrast(channel, min_val, max_val):
            channel = np.clip(channel, min_val, max_val)  # Clip values within range
            channel = 255 * (channel - min_val) / (max_val - min_val + 1e-8)  # Normalize and scale to 0-255
            return np.clip(channel, 0, 255)  # Ensure values remain in [0,255]

        frame_np[:, :, 0] = adjust_contrast(frame_np[:, :, 0], red_min_val, red_max_val)  # Red
        frame_np[:, :, 1] = adjust_contrast(frame_np[:, :, 1], green_min_val, green_max_val)  # Green

        # Convert back to uint8
        frame_np = frame_np.astype(np.uint8)

        if split_channels:
            # Create Red-only frame (Zero out Green & Blue)
            red_only_frame = frame_np.copy()
            red_only_frame[:, :, 1] = 0  # Remove Green
            red_only_frame[:, :, 2] = 0  # Remove Blue
            red_image = Image.fromarray(red_only_frame)

            # Create Green-only frame (Zero out Red & Blue)
            green_only_frame = frame_np.copy()
            green_only_frame[:, :, 0] = 0  # Remove Red
            green_only_frame[:, :, 2] = 0  # Remove Blue
            green_image = Image.fromarray(green_only_frame)

            # Resize both to 2x size
            red_image = red_image.resize((adjusted_frame.width * 2, adjusted_frame.height * 2))
            green_image = green_image.resize((adjusted_frame.width * 2, adjusted_frame.height * 2))

            # Combine Red and Green side-by-side into one image for synchronized playback
            combined_width = red_image.width + green_image.width
            combined_height = red_image.height
            combined_image = Image.new("RGB", (combined_width, combined_height))
            combined_image.paste(red_image, (0, 0))
            combined_image.paste(green_image, (red_image.width, 0))

            combined_frames.append(combined_image)

        else:
            frames.append(Image.fromarray(frame_np))

    if split_channels:
        # Save Combined Red-Green GIF for Synchronized Playback
        combined_gif_io = io.BytesIO()
        combined_frames[0].save(combined_gif_io, format="GIF", save_all=True, append_images=combined_frames[1:], loop=0, duration=frame_duration)
        combined_gif_io.seek(0)

        return combined_gif_io
    else:
        # Save Original Processed GIF
        gif_io = io.BytesIO()
        frames[0].save(gif_io, format="GIF", save_all=True, append_images=frames[1:], loop=0, duration=frame_duration)
        gif_io.seek(0)

        return gif_io

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
ROOT_DIR_FORWARD = './forward_fig'
ROOT_DIR_REVERSE = './reverse_fig'
ROOT_DIR_STRUCTURE = './structure_fig'
ROOT_DIR_RANDOM = './random_fig'

# Collect GIF files
def get_gif_files(root_dir):
    gif_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.gif'):
                gif_files.append(os.path.join(root, file))
    return gif_files

forward_gif_files = get_gif_files(ROOT_DIR_FORWARD)
forward_outline_gif_files = get_gif_files(ROOT_DIR_FORWARD+"_outlined")
reverse_gif_files = get_gif_files(ROOT_DIR_REVERSE)
reverse_outline_gif_files = get_gif_files(ROOT_DIR_REVERSE+"_outlined")
structure_gif_files = get_gif_files(ROOT_DIR_STRUCTURE)
structure_outline_gif_files = get_gif_files(ROOT_DIR_STRUCTURE+"_outlined")
random_gif_files = get_gif_files(ROOT_DIR_RANDOM)
random_outline_gif_files = get_gif_files(ROOT_DIR_RANDOM+"_outlined")

# Folders
folder_dict = {"forward": forward_gif_files,
               "reverse": reverse_gif_files,
               "+structure": structure_gif_files,
               "random": random_gif_files,
               "forward-outline": forward_outline_gif_files,
               "reverse-outline": reverse_outline_gif_files,
               "+structure-outline": structure_outline_gif_files,
               "random-outline": random_outline_gif_files,
            }

st.sidebar.header("Generation Methods")
use_method = st.sidebar.radio("Choose a generation method", ["forward selection", "reverse selection", "+structure", "random"])

st.sidebar.header("Display Options")
# Checkbox for choosing display mode (moved to the top)
display_split_gif = st.sidebar.checkbox("Side-by-side Display")
use_alt_folder = st.sidebar.checkbox("Add Outlines")

# gif_files = alt_gif_files if use_alt_folder else main_gif_files
gif_files = folder_dict[use_method.split(' ')[0] + ("-outline" if use_alt_folder else "")]

gif_options = {
    os.path.basename(os.path.dirname(gif)) + ' - Variant: ' + str(int(os.path.basename(gif).split('_')[0]) + 1): gif
    for gif in gif_files
}
# Stop at frame option
stop_at_frame = st.sidebar.checkbox("Stop at a specific frame")
stop_frame = st.sidebar.number_input("Frame to stop at (0-192)", min_value=0, max_value=192, step=1) if stop_at_frame else None


# Sliders for adjustments
st.sidebar.header("Brightness and Contrast")
brightness = st.sidebar.slider("Brightness", 0.1, 5.0, 1.0, 0.1)
contrast = st.sidebar.slider("Contrast", 0.1, 5.0, 1.0, 0.1)

# Contrast sliders for Red and Green channels
st.sidebar.header("Channel-wise Adjustment")
st.sidebar.subheader("Red Channel (Nucleus)")
red_intensity = st.sidebar.slider("Red Intensity", 0.0, 3.0, 1.0, 0.1)
red_min_val = st.sidebar.slider("Red Min Pixel Value", 0, 255, 0, 1)
red_max_val = st.sidebar.slider("Red Max Pixel Value", 0, 255, 255, 1)

st.sidebar.subheader("Green Channel (Structure)")
green_intensity = st.sidebar.slider("Green Intensity", 0.0, 3.0, 1.0, 0.1)
green_min_val = st.sidebar.slider("Green Min Pixel Value", 0, 255, 0, 1)
green_max_val = st.sidebar.slider("Green Max Pixel Value", 0, 255, 255, 1)

blue_intensity = 1  # Default, currently not user-controlled


#
st.sidebar.header("Playback Control")
gif_speed = st.sidebar.slider("Playback Speed (ms per frame)", 20, 100, 100, 5)

# Sidebar for GIF selection
st.sidebar.header("Select a GIF")
selected_gif_name = st.sidebar.radio("Choose a GIF", list(gif_options.keys()))

# Display selected GIF
if selected_gif_name:
    st.title(selected_gif_name)
    gif_path = gif_options[selected_gif_name]
    gene_name = selected_gif_name.split(' ')[0]
    description = get_gene_description(gene_name)
    st.write(f"**Description:** {description}")

    if display_split_gif:
        combined_gif = process_gif(
            gif_path, brightness, contrast, red_intensity, green_intensity, blue_intensity, 
            red_min_val, red_max_val, green_min_val, green_max_val, stop_frame, 
            split_channels=True, frame_duration=gif_speed
        )

        st.image(combined_gif.getvalue(), caption="Synchronized Red & Green GIF", use_container_width=True)
    else:
        # Show the original adjusted GIF
        processed_gif = process_gif(
            gif_path, brightness, contrast, red_intensity, green_intensity, blue_intensity, 
            red_min_val, red_max_val, green_min_val, green_max_val, stop_frame, 
            split_channels=False, frame_duration=gif_speed
        )
        st.image(processed_gif.getvalue(), caption=f"{selected_gif_name}", use_container_width=False, width=500)
