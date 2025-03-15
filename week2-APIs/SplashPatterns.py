import streamlit as st
import requests
import numpy as np
import cv2
from sklearn.cluster import KMeans
from skimage import color
from io import BytesIO
from PIL import Image

# ========== 1. 自定义CSS样式(仅保留侧边栏/按钮/标题的样式) ========== #
st.markdown("""
    <style>
    /* 给左侧Sidebar增加背景色与轻微阴影，以增强层次感 */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
        box-shadow: 2px 0 5px rgba(0, 0, 0, 0.1);
    }

    /* 主标题样式：加大字号、字母间距、减少底部外边距 */
    .main-title {
        font-size: 3rem;        
        letter-spacing: 2px;    
        font-weight: 700;
        margin-bottom: 0.5rem;  
        text-align: center;     
    }

    /* 侧边栏标题的样式 */
    [data-testid="stSidebar"] h2 {
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        letter-spacing: 1px;
    }

    /* 下拉框与按钮之间的默认间距 */
    .stSelectbox, .stButton {
        margin-bottom: 1rem;
    }

    /* 优化按钮外观与尺寸 */
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

# ========== 2. 自定义主标题 ========== #
st.markdown("<h1 class='main-title'>SplashPatterns</h1>", unsafe_allow_html=True)

# ========== 3. Unsplash API配置 ========== #
UNSPLASH_ACCESS_KEY = "cV2Slomwnm9YY0dp_lRF40J2QGatJfmmPxwJyzyZIlA"  # 请替换为你自己的 Access Key
UNSPLASH_SEARCH_URL = "https://api.unsplash.com/search/photos"

def search_unsplash_single_page(query, per_page=24, page=1, color=None):
    """
    在 Unsplash 上搜索单页的图片，返回 'regular' 图片 URL 列表。
    如果传入 color 参数，则会使用 Unsplash 的内置 color 筛选。
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
            st.warning(f"Unsplash API 返回错误: {response.status_code}")
            return []
        data = response.json()
    except Exception as e:
        st.error(f"请求 Unsplash API 出错: {e}")
        return []

    image_urls = []
    for item in data.get("results", []):
        url = item.get("urls", {}).get("regular")
        if url:
            image_urls.append(url)
    return image_urls

# ========== 4. 侧边栏输入和选择 ========== #
st.sidebar.header("Tone Selector")

# 用户仅能选择颜色与是否使用 K-Means 进行二次筛选
colors = [
    "None", "black_and_white", "black", "white", "yellow", "orange",
    "red", "purple", "magenta", "green", "teal", "blue"
]
selected_color = st.sidebar.selectbox("Select a color (Unsplash filter)", colors)
refine_color = st.sidebar.checkbox("Refine with K-Means")

# ========== 5. Session State 初始化 ========== #
if "images" not in st.session_state:
    st.session_state["images"] = []
if "page" not in st.session_state:
    st.session_state["page"] = 1

# ========== 6. 加载图片函数 ========== #
def load_images(color=None):
    """
    固定搜索关键词为 "pattern"，根据 session_state["page"] 获取图片，
    并追加到 session_state["images"] 中。
    """
    query = "pattern"
    new_urls = search_unsplash_single_page(
        query,
        per_page=24,
        page=st.session_state["page"],
        color=color
    )
    st.session_state["images"].extend(new_urls)

# ========== 7. 点击 "Show Patterns" 按钮 ========== #
if st.sidebar.button("Show Patterns"):
    # 每次新搜索都重置
    st.session_state["page"] = 1
    st.session_state["images"] = []
    
    c = None if selected_color == "None" else selected_color
    # 加载前 3 页数据
    for page_idx in range(1, 4):
        st.session_state["page"] = page_idx
        load_images(color=c)

# ========== 8. （可选）前端二次聚类色彩，用于精准筛选或排序 ========== #
if st.session_state["images"]:
    if refine_color and selected_color != "None":
        # 这里写一些近似 RGB 值；更专业的实现可以使用更详细的颜色映射
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
            """ 将 (r, g, b) [0,1] 转成 Lab """
            arr = np.array(rgb).reshape(1, 1, 3)
            lab = color.rgb2lab(arr)
            return lab[0, 0, :]

        target_lab = rgb_to_lab(target_rgb)

        def get_dominant_color(img, k=1):
            """
            使用 K-Means 从图片提取最主要的颜色（取聚类中心之一）。
            返回值为 (r, g, b)，取 0-1 区间。
            """
            img = np.array(img.resize((100, 100)))
            img = img.reshape(-1, 3).astype(np.float32) / 255.0
            km = KMeans(n_clusters=k, random_state=42).fit(img)
            labels, counts = np.unique(km.labels_, return_counts=True)
            major_cluster = labels[np.argmax(counts)]
            dominant_rgb = km.cluster_centers_[major_cluster]
            return dominant_rgb

        def color_distance(c1, c2):
            """ 计算两种 Lab 颜色的欧氏距离 """
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
                # 下载失败或处理出错时跳过该图片
                pass

        # 根据颜色距离从小到大排序，并只保留前 N 张图片
        refined_data.sort(key=lambda x: x[1])
        top_n = 200
        refined_data = refined_data[:top_n]
        st.session_state["images"] = [x[0] for x in refined_data]

    # ========== 9. 显示图片 + "Load More" 按钮 ========== #
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
