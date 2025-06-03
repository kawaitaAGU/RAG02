import streamlit as st
import base64
import io
from PIL import Image
import pandas as pd
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# APIキー取得
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# タイトル
st.title("画像から問題を解釈し、回答と解説を生成")

# ファイルアップロード
uploaded_file = st.file_uploader("問題画像をアップロードしてください", type=["jpg", "jpeg", "png"])

# RAG用のデータ（Excel想定）
@st.cache_data
def load_rag_data():
    return pd.read_excel("questions_db.xlsx")

# OCR: GPT-4oに画像で読み取りさせる
def extract_text_with_gpt(image_bytes):
    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    image_input = {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/png;base64,{base64_image}"
        }
    }

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "あなたは画像から日本語の問題文を正確に読み取り、読み取ったテキストのみを出力するOCR専門家です。"
            },
            {
                "role": "user",
                "content": [image_input]
            }
        ],
        max_tokens=1000
    )
    return response.choices[0].message.content.strip()

# 類似問題検索（TF-IDF + cosine 類似度）
def search_similar_questions(text, df, top_k=3):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df["question"])
    query_vec = vectorizer.transform([text])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[::-1][:top_k]
    return df.iloc[top_indices]

# GPTによる問題文の解釈・解説・回答
def explain_problem_with_gpt(problem_text, similar_questions_text):
    prompt = f"""次の日本語の試験問題に対して、以下を行ってください：
1. 問題全体の概要と意図の解説
2. 正解の選択肢とその理由
3. 各選択肢に対する個別の解説（正誤含む）

問題文：
{problem_text}

参考にすべき類似問題（過去問）：
{similar_questions_text}
"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "あなたは試験問題を解釈して、専門的かつわかりやすく解説する教育者です。"},
            {"role": "user", "content": prompt}
        ],
        max_tokens=2000
    )
    return response.choices[0].message.content.strip()

# メイン処理
if uploaded_file is not None:
    st.image(uploaded_file, caption="アップロード画像", use_column_width=True)

    # 画像をバイトに変換
    image = Image.open(uploaded_file)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    img_bytes = buffer.getvalue()

    with st.spinner("GPTが画像を解析しています（OCR）..."):
        extracted_text = extract_text_with_gpt(img_bytes)
        st.subheader("抽出された問題文")
        st.text_area("問題文", extracted_text, height=200)

    rag_df = load_rag_data()
    similar_df = search_similar_questions(extracted_text, rag_df)
    st.subheader("類似問題（RAG検索）")
    st.dataframe(similar_df)

    similar_text_concat = "\n\n".join(similar_df["question"].tolist())

    if st.button("GPTに解説を依頼"):
        with st.spinner("GPTが解説を生成中..."):
            explanation = explain_problem_with_gpt(extracted_text, similar_text_concat)
            st.subheader("GPTによる解説と回答")
            st.markdown(explanation)
