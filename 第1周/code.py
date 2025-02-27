import streamlit as st
import requests
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time
from datetime import datetime

# Streamlit 页面配置
st.set_page_config(page_title="Personalized Movie Recommendation", layout="wide")
st.title("Personalized Movie Recommendation System - Improved Version")

# TMDb API 配置
API_KEY = "9b27aa294bbc18b9c2a2d109f1e88402ç"  # 替换为你的 TMDb API Key
BASE_URL = "https://api.themoviedb.org/3"
IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"

def discover_movies(page=1):
    """
    从 /discover/movie 获取热门电影数据，每页约20部。
    """
    url = f"{BASE_URL}/discover/movie"
    params = {
        "api_key": API_KEY,
        "language": "en-US",
        "sort_by": "popularity.desc",
        "include_adult": "false",
        "include_video": "false",
        "page": page
    }
    response = requests.get(url, params=params).json()
    return response.get("results", [])

def search_movie(query):
    """
    从 /search/movie 根据用户输入的电影标题进行搜索，返回最匹配的电影（只取第一条）。
    如果没搜到，则返回 None。
    """
    url = f"{BASE_URL}/search/movie"
    params = {
        "api_key": API_KEY,
        "query": query,
        "language": "en-US"
    }
    response = requests.get(url, params=params).json()
    results = response.get("results", [])
    return results[0] if results else None

# 获取多页电影数据（示例获取前5页，每页约20部，共~100部电影）
all_movies = []
for page in range(1, 6):
    movies_page = discover_movies(page=page)
    all_movies.extend(movies_page)
    time.sleep(0.2)  # 防止请求过快

# 构建 DataFrame
movies_list = []
for movie in all_movies:
    movies_list.append({
        "title": movie.get("title", ""),
        "overview": movie.get("overview", ""),
        "poster_path": movie.get("poster_path"),
        "vote_average": movie.get("vote_average", 0.0)
    })
df_movies = pd.DataFrame(movies_list)

st.write("Fetched {} movies in total.".format(len(df_movies)))
st.write(df_movies.head())

# ---------- 改进相似度算法思路 ----------
# 目前我们仍然使用 TF-IDF 对 overview 进行向量化，然后做余弦相似度。
# 可扩展为：
#   1) CountVectorizer + TfidfTransformer
#   2) 预训练语言模型 (Sentence-BERT / GPT embedding)
#   3) 多特征 (genres + overview + cast 等) 拼接
# 此处示例仍使用简单 TF-IDF。
# --------------------------------------

# 构建 TF-IDF 矩阵
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df_movies['overview'].fillna(""))

def recommend_by_text(query_text, top_n=5):
    """
    用用户输入的文本与电影简介做 TF-IDF 相似度，返回最相似的 top_n 电影。
    """
    query_vec = tfidf.transform([query_text])
    sim_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = sim_scores.argsort()[-top_n:][::-1]
    return df_movies.iloc[top_indices], sim_scores[top_indices]

# Streamlit 交互
st.sidebar.header("Recommendation Settings")
user_input = st.sidebar.text_input("Enter a movie title or keywords", "")
top_n = st.sidebar.slider("Number of Recommendations", 1, 20, 5)

if user_input:
    st.header(f"Recommendations for: {user_input}")

    # 1) 优先尝试搜索是否是某部已存在的电影标题
    found_movie = search_movie(user_input)
    if found_movie:
        # 如果找到，就用该电影的 overview 作为查询
        query_text = found_movie.get("overview", "")
        st.write(f"Found a movie match: **{found_movie.get('title')}**")
    else:
        # 如果没搜到，就直接把用户输入当作关键词
        query_text = user_input
        st.write("No exact movie found by title. Using input text as keywords...")

    # 2) 调用 recommend_by_text
    recs, scores = recommend_by_text(query_text, top_n=top_n)

    # 3) 显示推荐结果
    for idx, row in recs.iterrows():
        col1, col2 = st.columns([1, 3])
        with col1:
            if pd.notna(row['poster_path']):
                st.image(IMAGE_BASE_URL + row['poster_path'], width=100)
            else:
                st.text("No Image")
        with col2:
            st.subheader(row['title'])
            st.write("Rating: {:.1f}".format(row['vote_average'] if row['vote_average'] else 0.0))
            st.write(row['overview'])
            sim_score = scores[list(recs.index).index(idx)]
            st.write("Similarity Score: {:.2f}".format(sim_score))

else:
    st.write("Please enter a movie title or keywords in the sidebar to get recommendations.")
