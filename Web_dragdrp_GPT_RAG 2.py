import streamlit as st
import base64
import io
import pandas as pd
from PIL import Image
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# === OpenAIクライアント初期化（Secretsから） ============================
if "OPENAI_API_KEY" not in st.secrets:
    st.error("OPENAI_API_KEY が未設定です。StreamlitのSecretsに追加してください。")
    st.stop()

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# === タイトルとアップロードUI ============================================
st.title("画像から問題を読み取り、過去問と照合して解説を生成")

uploaded_img = st.file_uploader("問題画像（.png, .jpg）をアップロード", type=["png", "jpg", "jpeg"])
uploaded_excel = st.file_uploader("過去問Excelファイル（.xlsx）をアップロード", type=["xlsx"])

if uploaded_img and uploaded_excel:
    # === 画像→Base64エンコード ==========================================
    image = Image.open(uploaded_img).convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    b64_img = base64.b64encode(buffer.getvalue()).decode()
    data_uri = f"data:image/png;base64,{b64_img}"

    # === GPT OCRで問題文抽出 ============================================
    with st.spinner("画像から問題文を抽出しています（GPT Vision）..."):
        extract_response = client.chat.completions.create(
            model="gpt-4o-2024-11-20",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_uri}},
                        {"type": "text", "text": "この画像に書かれている問題文を読み取ってください。"}
                    ]
                }
            ],
            max_tokens=2000,
            temperature=0.0
        )
        query_text = extract_response.choices[0].message.content.strip()
        st.subheader("🔍 抽出された問題文")
        st.text_area("問題文", query_text, height=200)

    # === Excel読み込みと過去問抽出 =====================================
    with st.spinner("Excelから類似問題を検索中（TF-IDF）..."):
        df = pd.read_excel(uploaded_excel)

        corpus = []
        index_to_row = []
        for i, row in df.iterrows():
            for cell in row:
                if isinstance(cell, str) and len(cell) > 10:
                    corpus.append(cell)
                    index_to_row.append(i)

        if not corpus:
            st.error("Excelから問題文候補が見つかりませんでした。")
            st.stop()

        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(corpus + [query_text])
        similarities = cosine_similarity(X[-1], X[:-1])[0]
        top_indices = similarities.argsort()[-3:][::-1]

        similar_questions = []
        for idx in top_indices:
            row = df.iloc[index_to_row[idx]]
            text = corpus[idx]
            choices = [str(cell) for cell in row if isinstance(cell, str) and 5 < len(cell) < 100 and cell != text]
            if not choices:
                choices = ["（選択肢情報が取得できませんでした）"]

            correct = ""
            for cell in row:
                if isinstance(cell, str) and cell.strip().upper() in ['A', 'B', 'C', 'D', 'E']:
                    correct = cell.strip().upper()
                    break

            qinfo = f"{text}\n選択肢:\n" + "\n".join(f"- {c}" for c in choices[:5])
            if correct:
                qinfo += f"\n正解と思われる選択肢: {correct}"
            similar_questions.append(qinfo)

        rag_text = "\n\n" + "\n\n".join(similar_questions)
        st.subheader("📚 類似問題（RAG）")
        for q in similar_questions:
            st.markdown(f"```\n{q}\n```")

    # === GPT解説ボタン ================================================
    if st.button("📘 GPTに解説を依頼する"):
        with st.spinner("GPTが問題の解釈と解説を生成中..."):
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": data_uri}},
                            {"type": "text", "text":
                                f"今送った画像の問題の解説をしてください。正解を明示し、根拠を説明してください。\n\n"
                                f"以下は過去問から抽出した類似問題情報です：{rag_text}"
                            }
                        ]
                    }
                ],
                max_tokens=2000,
                temperature=0.1
            )
            st.subheader("💡 GPTの解説結果")
            st.markdown(response.choices[0].message.content.strip())
