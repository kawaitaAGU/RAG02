import streamlit as st
import base64
import io
from PIL import Image
import pandas as pd
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# OpenAI APIキーをStreamlit secretsから取得（存在チェック付き）
if "OPENAI_API_KEY" not in st.secrets:
    st.error("OPENAI_API_KEY が設定されていません。Streamlit Cloud の Secrets に登録してください。")
    st.stop()

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# 画像アップロードUI
st.title("画像アップロードで問題文を抽出・解説")
uploaded_file = st.file_uploader("問題画像をアップロードしてください", type=["jpg", "jpeg", "png"])

# RAG用データ読み込み（Excel想定）
@st.cache_data
def load_rag_data():
    df = pd.read_excel("questions_db.xlsx")  # Excel読み込みには openpyxl が必要
    return df

# 類似問題検索（TF-IDF + cosine類似度）
def search_similar_questions(text, df, top_k=3):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df["question"])
    query_vec = vectorizer.transform([text])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[::-1][:top_k]
    return df.iloc[top_indices]

# GPTで解説生成
def explain_with_gpt(prompt_text):
    response = client.chat.completions.create(
        model="gpt-4o-2024-11-20",
        messages=[
            {"role": "system", "content": "あなたは試験問題の解説者です。わかりやすく解説してください。"},
            {"role": "user", "content": prompt_text}
        ]
    )
    return response.choices[0].message.content.strip()

# メイン処理
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="アップロードされた画像", use_column_width=True)

    # OCRは未実装：仮にbase64変換だけ表示
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    img_bytes = buffer.getvalue()
    base64_img = base64.b64encode(img_bytes).decode()
    st.text("Base64形式の画像（例）:")
    st.text_area("base64", base64_img[:300] + "...", height=100)

    # 画像からの問題文抽出（手入力で仮定）
    problem_text = st.text_area("抽出された問題文（例）", "例：次の中で正しいものを1つ選べ。")

    if st.button("GPTに説明を依頼"):
        with st.spinner("GPTが解説中..."):
            explanation = explain_with_gpt(problem_text)
            st.subheader("GPTの解説:")
            st.write(explanation)

            st.subheader("関連する類似問題（RAG）:")
            rag_df = load_rag_data()
            similar = search_similar_questions(problem_text, rag_df)
            st.dataframe(similar)
