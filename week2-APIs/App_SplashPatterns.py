import streamlit as st
import requests
import numpy as np
import cv2
from sklearn.cluster import KMeans
from skimage import color
from io import BytesIO
from PIL import Image

# ========== 1. Custom CSS Styling (only for sidebar/buttons/titles) ========== #
st.markdown("""
    <style>
    /* Add a background color and subtle shadow to the left sidebar to enhance layering */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
        box-shadow: 2px 0 5px rgba(0, 0, 0, 0.1);
    }

    /* Main title style: larger font size, increased letter spacing, and reduced bottom margin */
    .main-title {
        font-size: 3rem;        
        letter-spacing: 2px;    
        font-weight: 700;
        margin-bottom: 0.5rem;  
        text-align: center;     
    }

    /* Sidebar title style */
    [data-testid="stSidebar"] h2 {
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        letter-spacing: 1px;
    }

    /* Default spacing between dropdown and button */
    .stSelectbox, .stButton {
        margin-bottom: 1rem;
    }

    /* Optimize button appearance and size */
    .stButton > button {
        font-size: 16px;
        padding: 0.5rem 1.5rem; 
        border-radius: 4px;
        background-color: #cccccc; 
        color: #333;               
        border: none;
        transition: background-color 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #bbbbbb;
    }
    </style>
""", unsafe_allow_html=True)

# ========== 2. Custom Main Title ========== #
st.markdown("<h1 class='main-title'>SplashPatterns</h1>", unsafe_allow_html=True)

# ========== 3. Unsplash API Configuration ========== #
UNSPLASH_ACCESS_KEY = "cV2Slomwnm9YY0dp_lRF40J2QGatJfmmPxwJyzyZIlA"  # Please replace with your own Access Key
UNSPLASH_SEARCH_URL = "https://api.unsplash.com/search/photos"

def search_unsplash_single_page(query, per_page=24, page=1, color=None):
    """
    Search for a single page of images on Unsplash, returning a list of 'regular' image URLs.
    If the color parameter is provided, the Unsplash built-in color filter will be used.
    """
    headers = {"Authorization": f"Client-ID {UNSPLASH_ACCESS_KEY}"}
    params = {
        "query": query,
        "page": page,
        "per_page": per_page
    }
    if color:
        params["color"] = color

    try:
        response = requests.get(UNSPLASH_SEARCH_URL, headers=headers, params=params)
        if response.status_code != 200:
            st.warning(f"Unsplash API returned error: {response.status_code}")
            return []
        data = response.json()
    except Exception as e:
        st.error(f"Error occurred while requesting Unsplash API: {e}")
        return []

    image_urls = []
    for item in data.get("results", []):
        url = item.get("urls", {}).get("regular")
        if url:
            image_urls.append(url)
    return image_urls

# ========== 4. Sidebar Input and Selection ========== #
st.sidebar.header("Tone Selector")

# Users can only choose a color and whether to refine using K-Means for secondary filtering
colors = [
    "None", "black_and_white", "black", "white", "yellow", "orange",
    "red", "purple", "magenta", "green", "teal", "blue"
]
selected_color = st.sidebar.selectbox("Select a color (Unsplash filter)", colors)
refine_color = st.sidebar.checkbox("Refine with K-Means")

# ========== 5. Initialize Session State ========== #
if "images" not in st.session_state:
    st.session_state["images"] = []
if "page" not in st.session_state:
    st.session_state["page"] = 1

# ========== 6. Function to Load Images ========== #
def load_images(color=None):
    """
    Use the fixed search keyword "pattern", retrieve images based on session_state["page"],
    and append them to session_state["images"].
    """
    query = "pattern"
    new_urls = search_unsplash_single_page(
        query,
        per_page=24,
        page=st.session_state["page"],
        color=color
    )
    st.session_state["images"].extend(new_urls)

# ========== 7. Click the "Show Patterns" Button ========== #
if st.sidebar.button("Show Patterns"):
    # Reset session state on each new search
    st.session_state["page"] = 1
    st.session_state["images"] = []
    
    c = None if selected_color == "None" else selected_color
    # Load data from the first 3 pages
    for page_idx in range(1, 4):
        st.session_state["page"] = page_idx
        load_images(color=c)

# ========== 8. Frontend Color Refinement using Clustering for Precise Filtering or Sorting ========== #
if st.session_state["images"]:
    if refine_color and selected_color != "None":
        # Approximate RGB values; a more professional implementation might use a more detailed color mapping
        color_map = {
            "black": (0, 0, 0),
            "white": (1, 1, 1),
            "yellow": (1, 1, 0),
            "orange": (1, 0.65, 0),
            "red": (1, 0, 0),
            "purple": (0.5, 0, 0.5),
            "magenta": (1, 0, 1),
            "green": (0, 1, 0),
            "teal": (0, 0.5, 0.5),
            "blue": (0, 0, 1),
            "black_and_white": (0.5, 0.5, 0.5)
        }
        target_rgb = color_map.get(selected_color, (0.5, 0.5, 0.5))
        def rgb_to_lab(rgb):
            """ Convert (r, g, b) in [0,1] to Lab """
            arr = np.array(rgb).reshape(1, 1, 3)
            lab = color.rgb2lab(arr)
            return lab[0, 0, :]

        target_lab = rgb_to_lab(target_rgb)

        def get_dominant_color(img, k=1):
            """
            Use K-Means to extract the dominant color from the image (using one of the cluster centers).
            Returns a tuple (r, g, b) in the range [0,1].
            """
            img = np.array(img.resize((100, 100)))
            img = img.reshape(-1, 3).astype(np.float32) / 255.0
            km = KMeans(n_clusters=k, random_state=42).fit(img)
            labels, counts = np.unique(km.labels_, return_counts=True)
            major_cluster = labels[np.argmax(counts)]
            dominant_rgb = km.cluster_centers_[major_cluster]
            return dominant_rgb

        def color_distance(c1, c2):
            """ Calculate the Euclidean distance between two Lab colors """
            return np.sqrt(np.sum((c1 - c2) ** 2))

        refined_data = []
        for url in st.session_state["images"]:
            try:
                resp = requests.get(url, timeout=5)
                if resp.status_code == 200:
                    img_pil = Image.open(BytesIO(resp.content)).convert("RGB")
                    dom_rgb = get_dominant_color(img_pil, k=3)
                    dist = color_distance(rgb_to_lab(dom_rgb), target_lab)
                    refined_data.append((url, dist))
            except Exception as e:
                # Skip the image if downloading fails or an error occurs
                pass

        # Sort images by color distance in ascending order and keep only the top N images
        refined_data.sort(key=lambda x: x[1])
        top_n = 200
        refined_data = refined_data[:top_n]
        st.session_state["images"] = [x[0] for x in refined_data]

    # ========== 9. Display Images and "Load More" Button ========== #
    num_cols = 4
    cols = st.columns(num_cols)
    for idx, url in enumerate(st.session_state["images"]):
        with cols[idx % num_cols]:
            st.image(url, use_container_width=True)

    if st.button("Load More"):
        st.session_state["page"] += 1
        c = None if selected_color == "None" else selected_color
        load_images(color=c)
elif st.session_state["page"] == 1:
    st.write("No images to display. Please click 'Show Patterns' on the sidebar.")
