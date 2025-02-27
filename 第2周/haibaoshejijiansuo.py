import streamlit as st
import requests

# 设置页面标题和布局
st.set_page_config(page_title="Pattern Search", layout="wide")
st.title("Pattern Search Dashboard")

# Unsplash API配置
UNSPLASH_ACCESS_KEY = "你的Access Key"  # 请替换为你自己的Access Key
UNSPLASH_SEARCH_URL = "https://api.unsplash.com/search/photos"

def search_unsplash(query, per_page=12):
    """
    使用Unsplash API搜索图片
    返回一个'regular'图片URL的列表
    同时在Streamlit中打印一些调试信息
    """
    headers = {"Authorization": f"Client-ID {UNSPLASH_ACCESS_KEY}"}
    params = {
        "query": query,
        "per_page": per_page
    }
    response = requests.get(UNSPLASH_SEARCH_URL, headers=headers, params=params)
    
    # 调试：打印状态码和返回JSON的关键结构
    st.write("DEBUG: status code =", response.status_code)
    data = response.json()
    st.write("DEBUG: data keys:", data.keys())
    
    if "results" not in data:
        st.write("DEBUG: full data:", data)
        return []
    
    results = data["results"]
    st.write("DEBUG: found", len(results), "items for query:", query)
    
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
# 无论如何都先搜索 "pattern"
final_query = "pattern"
if selected_color != "None":
    # 如果选择了具体颜色，则在查询后面追加
    final_query += f" {selected_color}"

# 在侧边栏显示一下最终的搜索关键词，便于调试
st.sidebar.write("Search Query:", final_query)

# 当用户点击按钮时，进行API搜索
if st.sidebar.button("Search Patterns"):
    st.write(f"Searching for patterns with query: **{final_query}**")
    image_urls = search_unsplash(final_query, per_page=12)
    if not image_urls:
        st.write("No results found. Please try different filters or check your API key.")
    else:
        # 将图片4列展示
        cols = st.columns(4)
        for idx, url in enumerate(image_urls):
            with cols[idx % 4]:
                st.image(url, use_container_width=True)
else:
    st.write("Please choose a color (optional) in the sidebar and click 'Search Patterns' to begin.")
