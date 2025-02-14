import streamlit as st
from PIL import Image, ImageEnhance
import io
import os
import numpy as np
import requests
import base64
import streamlit.components.v1 as components

#####################################
# Helper Functions for File Handling#
#####################################

def get_subfolder_names(parent_dir):
    """Return a list of full paths to subfolders within parent_dir."""
    try:
        subfolders = [entry.path for entry in os.scandir(parent_dir) if entry.is_dir()]
        return subfolders
    except Exception as e:
        return []

def get_gif_files(directory):
    """Return a list of GIF file paths from the given directory (including subdirectories)."""
    gif_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.gif'):
                gif_files.append(os.path.join(root, file))
    return gif_files

#####################################
# GIF Processing Function           #
#####################################

def process_gif(gif_path, brightness, contrast, red_intensity, green_intensity, blue_intensity, 
                red_min_val, red_max_val, green_min_val, green_max_val, stop_frame=None, 
                split_channels=False, frame_duration=100):
    """
    Process the GIF (or its frames) with the specified parameters.
    
    If stop_frame is provided, only that frame is processed.
    """
    gif = Image.open(gif_path)
    frames = []
    combined_frames = []

    for frame in range(gif.n_frames):
        if stop_frame is not None:
            if frame < stop_frame:
                continue
            elif frame > stop_frame:
                break
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

        # Convert to NumPy array for additional contrast adjustment
        frame_np = np.array(adjusted_frame, dtype=np.float32)

        def adjust_contrast(channel, min_val, max_val):
            channel = np.clip(channel, min_val, max_val)
            channel = 255 * (channel - min_val) / (max_val - min_val + 1e-8)
            return np.clip(channel, 0, 255)
        
        frame_np[:, :, 0] = adjust_contrast(frame_np[:, :, 0], red_min_val, red_max_val)
        frame_np[:, :, 1] = adjust_contrast(frame_np[:, :, 1], green_min_val, green_max_val)

        frame_np = frame_np.astype(np.uint8)

        if split_channels:
            # Create red-only and green-only images
            red_only_frame = frame_np.copy()
            red_only_frame[:, :, 1] = 0
            red_only_frame[:, :, 2] = 0
            red_image = Image.fromarray(red_only_frame)

            green_only_frame = frame_np.copy()
            green_only_frame[:, :, 0] = 0
            green_only_frame[:, :, 2] = 0
            green_image = Image.fromarray(green_only_frame)

            # Resize both to 2x size
            red_image = red_image.resize((adjusted_frame.width * 2, adjusted_frame.height * 2))
            green_image = green_image.resize((adjusted_frame.width * 2, adjusted_frame.height * 2))

            # Combine side-by-side
            combined_width = red_image.width + green_image.width
            combined_height = red_image.height
            combined_image = Image.new("RGB", (combined_width, combined_height))
            combined_image.paste(red_image, (0, 0))
            combined_image.paste(green_image, (red_image.width, 0))
            combined_frames.append(combined_image)
        else:
            frames.append(Image.fromarray(frame_np))

        if stop_frame is not None:
            break  # only process the designated frame

    # Save as animated GIF if multiple frames; otherwise, as static PNG.
    out_io = io.BytesIO()
    if split_channels:
        if len(combined_frames) > 1:
            combined_frames[0].save(out_io, format="GIF", save_all=True,
                                      append_images=combined_frames[1:], loop=0, duration=frame_duration)
        else:
            combined_frames[0].save(out_io, format="PNG")
    else:
        if len(frames) > 1:
            frames[0].save(out_io, format="GIF", save_all=True,
                           append_images=frames[1:], loop=0, duration=frame_duration)
        else:
            frames[0].save(out_io, format="PNG")
    out_io.seek(0)
    return out_io

#####################################
# Gene Description Function         #
#####################################

def get_gene_description(gene_name):
    """Fetch a gene description from the SGD API."""
    base_url = f"https://www.yeastgenome.org/backend/locus/{gene_name}"
    response = requests.get(base_url)
    if response.status_code == 200:
        data = response.json()
        return data.get("description", "No description available")
    else:
        return f"Error: Unable to fetch data for {gene_name} (status {response.status_code})"

#####################################
# Base Folder Setup                 #
#####################################

ROOT_DIR_FORWARD = './forward_fig'
ROOT_DIR_REVERSE = './reverse_fig'
ROOT_DIR_STRUCTURE = './structure_fig'
ROOT_DIR_RANDOM = './random_fig'

def get_base_folder(gen_method, use_outline=False):
    """Return the base folder (string path) for a given generation method."""
    if gen_method == "forward selection":
        base = ROOT_DIR_FORWARD
    elif gen_method == "reverse selection":
        base = ROOT_DIR_REVERSE
    elif gen_method == "+structure":
        base = ROOT_DIR_STRUCTURE
    elif gen_method == "random":
        base = ROOT_DIR_RANDOM
    else:
        base = ""
    if use_outline:
        base += "_outlined"
    return base

#####################################
# Sidebar Configuration             #
#####################################

st.sidebar.header("Display Mode")
display_mode = st.sidebar.radio("Choose Display Mode", ["Single Image", "Multiple Images"])

st.sidebar.header("Generation Methods")
if display_mode == "Single Image":
    use_method = st.sidebar.radio("Choose a generation method", 
                                  ["forward selection", "reverse selection", "+structure", "random"])

st.sidebar.header("Display Options")
display_split_gif = st.sidebar.checkbox("Side-by-side Display")
use_alt_folder = st.sidebar.checkbox("Add Outlines")

#####################################
# Other Adjustments                 #
#####################################

stop_at_frame = st.sidebar.checkbox("Stop at a specific frame")
stop_frame = st.sidebar.number_input("Frame to stop at (0-192)", min_value=0, max_value=192, step=1) if stop_at_frame else None

st.sidebar.header("Brightness and Contrast")
brightness = st.sidebar.slider("Brightness", 0.1, 5.0, 1.0, 0.1)
contrast = st.sidebar.slider("Contrast", 0.1, 5.0, 1.0, 0.1)

st.sidebar.header("Channel-wise Adjustment")
st.sidebar.subheader("Red Channel (Nucleus)")
red_intensity = st.sidebar.slider("Red Intensity", 0.0, 3.0, 1.0, 0.1)
red_min_val = st.sidebar.slider("Red Min Pixel Value", 0, 255, 0, 1)
red_max_val = st.sidebar.slider("Red Max Pixel Value", 0, 255, 255, 1)

st.sidebar.subheader("Green Channel (Structure)")
green_intensity = st.sidebar.slider("Green Intensity", 0.0, 3.0, 1.0, 0.1)
green_min_val = st.sidebar.slider("Green Min Pixel Value", 0, 255, 0, 1)
green_max_val = st.sidebar.slider("Green Max Pixel Value", 0, 255, 255, 1)

blue_intensity = 1  # Fixed

st.sidebar.header("Playback Control")
gif_speed = st.sidebar.slider("Playback Speed (ms per frame)", 20, 100, 100, 5)

#####################################
# Subfolder Selection for Single Image (Sidebar)
#####################################

if display_mode == "Single Image":
    base_folder = get_base_folder(use_method, use_alt_folder)
    subfolders = get_subfolder_names(base_folder)
    if subfolders:
        subfolder_names = [os.path.basename(sf) for sf in subfolders]
        selected_subfolder_name = st.sidebar.selectbox("Select a Subfolder", subfolder_names)
        selected_subfolder = [sf for sf in subfolders if os.path.basename(sf) == selected_subfolder_name][0]
    else:
        st.sidebar.write("No subfolders found in", base_folder)
        selected_subfolder = base_folder

#####################################
# Single Image Mode                 #
#####################################

if display_mode == "Single Image":
    st.title("Single Image Display")
    gif_files = get_gif_files(selected_subfolder)
    if not gif_files:
        st.write("No GIF files found in", selected_subfolder)
    else:
        gif_options = {
            os.path.basename(os.path.dirname(gif)) + ' - Variant: ' + str(int(os.path.basename(gif).split('_')[0]) + 1): gif
            for gif in gif_files
        }
        st.sidebar.header("Select a GIF")
        selected_gif_name = st.sidebar.radio("Choose a GIF", list(gif_options.keys()))
        if selected_gif_name:
            st.header(selected_gif_name)
            gif_path = gif_options[selected_gif_name]
            gene_name = selected_gif_name.split(' ')[0]
            description = get_gene_description(gene_name)
            st.write(f"**Description:** {description}")
            
            effective_stop_frame = stop_frame
            
            processed = process_gif(
                gif_path, brightness, contrast, red_intensity, green_intensity, blue_intensity,
                red_min_val, red_max_val, green_min_val, green_max_val, effective_stop_frame,
                split_channels=display_split_gif, frame_duration=gif_speed
            )
            st.image(processed.getvalue(), caption=selected_gif_name, width=500)

#####################################
# Multiple Images Mode              #
#####################################

else:
    st.title("Multiple Images Display")

    # Sidebar: For each generation method, let the user select a subfolder.
    st.sidebar.header("Subfolder Selection for Multiple Images")
    
    method_info = {
        "Forward": "forward selection",
        "Reverse": "reverse selection",
        "+Structure": "+structure",
        "Random": "random"
    }
    selected_folders = {}
    for label, method in method_info.items():
        base_dir = get_base_folder(method, use_alt_folder)
        subfolders = get_subfolder_names(base_dir)
        if subfolders:
            subfolder_names = [os.path.basename(sf) for sf in subfolders]
            selected_name = st.sidebar.selectbox(f"{label} Subfolder", subfolder_names, key=label)
            selected_folder = [sf for sf in subfolders if os.path.basename(sf)==selected_name][0]
        else:
            selected_folder = base_dir
        selected_folders[label] = selected_folder

    # Build one combined HTML gallery.
    gallery_html = """
    <html>
      <head>
        <style>
            body { 
                color: white; 
                background-color: #000; /* optional, if you want a dark background */
            }
            .gallery-container { 
                display: flex; 
                flex-wrap: wrap; 
            }
            .gallery-item { 
                flex: 1; 
                margin: 2px; 
                max-width: 300px; 
            }
            .gallery-item img { 
                width: 100%; 
            }
            .gallery-item p { 
                text-align: center; 
                font-size: 0.8em; 
                margin: 2px 0; 
                color: white; /* white font for captions */
            }
            h3 { 
                margin-top: 20px; 
                margin-bottom: 5px; 
                color: white; /* white font for headers */
            }
            button { 
                padding:10px; 
                font-size:1em; 
                margin-top: 10px; 
            }
        </style>
        <script>
          function restartGifs() {
            var gifs = document.getElementsByClassName('sync-gif');
            for (var i = 0; i < gifs.length; i++) {
              var src = gifs[i].src;
              gifs[i].src = "";
              gifs[i].src = src;
            }
          }
          window.onload = restartGifs;
        </script>
      </head>
      <body>
    """
    # For each method, add a row with header and gallery items.
    for label, folder in selected_folders.items():
        gif_files = get_gif_files(folder)
        if not gif_files:
            gallery_html += f"<h3>{label} ({os.path.basename(folder)})</h3><p>No GIF files found.</p>"
            continue
        gallery_html += f"<h3>{label} ({os.path.basename(folder)})</h3><div class='gallery-container'>"
        for gif_path in gif_files:
            if display_split_gif:
                processed = process_gif(
                    gif_path, brightness, contrast, red_intensity, green_intensity, blue_intensity,
                    red_min_val, red_max_val, green_min_val, green_max_val, None,
                    split_channels=True, frame_duration=gif_speed
                )
            else:
                processed = process_gif(
                    gif_path, brightness, contrast, red_intensity, green_intensity, blue_intensity,
                    red_min_val, red_max_val, green_min_val, green_max_val, None,
                    split_channels=False, frame_duration=gif_speed
                )
            b64 = base64.b64encode(processed.getvalue()).decode("utf-8")
            gallery_html += f"""
                <div class='gallery-item'>
                  <img class='sync-gif' src="data:image/gif;base64,{b64}" />
                  <p>{os.path.basename(gif_path)}</p>
                </div>
            """
        gallery_html += "</div>"
    gallery_html += """
        <button onclick="restartGifs()">Restart Animations</button>
      </body>
    </html>
    """
    components.html(gallery_html, height=800, scrolling=True)
