import streamlit as st
import requests

# 设置页面标题和布局
st.set_page_config(page_title="UnPatterned", layout="wide")
st.title("UnPatterned")

# Unsplash API配置
UNSPLASH_ACCESS_KEY = "cV2Slomwnm9YY0dp_lRF40J2QGatJfmmPxwJyzyZIlA"  # 请替换为你自己的Access Key
UNSPLASH_SEARCH_URL = "https://api.unsplash.com/search/photos"

def search_unsplash(query, per_page=12):
    """
    使用Unsplash API搜索图片
    返回一个'regular'图片URL的列表
    """
    headers = {"Authorization": f"Client-ID {UNSPLASH_ACCESS_KEY}"}
    params = {
        "query": query,
        "per_page": per_page
    }
    response = requests.get(UNSPLASH_SEARCH_URL, headers=headers, params=params)
    
    # 尝试解析为JSON
    data = response.json()
    
    # 如果没有"results"，返回空列表
    if "results" not in data:
        return []
    
    # 提取图片URL
    results = data["results"]
    image_urls = []
    for item in results:
        url = item.get("urls", {}).get("regular")
        if url:
            image_urls.append(url)
    return image_urls

# 侧边栏：提供颜色选择（或者你可以换成其他分类方式）
st.sidebar.header("Filter Options")

colors = ["None", "Red", "Blue", "Green", "Yellow", "Black", "White", "Purple", "Gray"]
selected_color = st.sidebar.selectbox("Choose a color (optional)", colors)

# 构建最终查询字符串
final_query = "pattern"
if selected_color != "None":
    final_query += f" {selected_color}"

# 按钮只显示"Search"
if st.sidebar.button("Search"):
    image_urls = search_unsplash(final_query, per_page=12)
    if not image_urls:
        st.write("No results found. Please try a different color or check your API key.")
    else:
        # 将图片4列展示
        cols = st.columns(4)
        for idx, url in enumerate(image_urls):
            with cols[idx % 4]:
                st.image(url, use_container_width=True)
