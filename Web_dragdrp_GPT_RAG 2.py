#gpt-4o-2024-11-20
import streamlit as st
import base64
import io
import pandas as pd
from PIL import Image
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from pathlib import Path

# === OpenAI APIキーの初期化 ========================================
if "OPENAI_API_KEY" not in st.secrets:
    st.error("OPENAI_API_KEY が未設定です。StreamlitのSecretsに追加してください。")
    st.stop()

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# === UIタイトルと画像アップロード欄 ===============================
st.title("画像から問題を読み取り、RAG付きで自動解説")

uploaded_img = st.file_uploader("問題画像をアップロード（.png, .jpg）", type=["png", "jpg", "jpeg"])

if uploaded_img:
    st.session_state.clear()

    st.image(uploaded_img, caption="アップロードされた画像", use_column_width=True)

    image = Image.open(uploaded_img).convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    st.session_state['b64_img'] = base64.b64encode(buffer.getvalue()).decode()

    data_uri = f"data:image/png;base64,{st.session_state['b64_img']}"

    # === GPTでOCR（画像→テキスト） ===============================
    with st.spinner("画像から問題文を抽出中..."):
        extract_response = client.chat.completions.create(
            model="gpt-4o-2024-11-20",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_uri}},
                        {"type": "text", "text": "この画像に書かれている問題文と選択肢を読み取ってください。"}
                    ]
                }
            ],
            max_tokens=2000,
            temperature=0.0
        )
        query_text = extract_response.choices[0].message.content.strip()

    # === OCR結果の表示（参考） =====================================
    with st.expander("📝 画像から読み取られた問題文・選択肢（OCR結果）", expanded=False):
        st.markdown("以下は、画像から抽出された問題文と選択肢のテキストです（参考表示）。")
        st.code(query_text, language="markdown")

    # === sample.csv を読み込んでRAG処理 ==========================
    rag_text = ""
    csv_path = Path("sample.csv")
    if not csv_path.exists():
        st.error("sample.csv が見つかりません。ファイルをアプリと同じフォルダに配置してください。")
        st.stop()

    try:
        with st.spinner("過去問データから類似問題を検索中..."):
            df = pd.read_csv(csv_path)

            if "問題文" not in df.columns:
                st.error("CSVファイルに '問題文' 列が含まれていません。")
                st.stop()

            corpus = df["問題文"].fillna("").tolist()
            if not corpus or len(query_text.strip()) < 10:
                st.warning("抽出された問題文が短すぎるか、空です。RAG検索をスキップします。")
            else:
                vectorizer = TfidfVectorizer()
                X = vectorizer.fit_transform(corpus + [query_text])
                similarities = cosine_similarity(X[-1], X[:-1])[0]
                top_indices = similarities.argsort()[-3:][::-1]

                similar_questions = []
                for idx in top_indices:
                    row = df.iloc[idx]
                    qtext = row["問題文"]
                    choices = [str(row[c]) for c in ['a', 'b', 'c', 'd', 'e'] if c in row and pd.notna(row[c])]
                    correct = str(row["解答"]) if "解答" in row and pd.notna(row["解答"]) else ""

                    qinfo = f"{qtext}\n選択肢:\n" + "\n".join([f"- {c}" for c in choices])
                    if correct:
                        qinfo += f"\n正解と思われる選択肢: {correct}"
                    similar_questions.append(qinfo)

                rag_text = "\n\n".join(similar_questions)
                st.subheader("📚 類似問題（RAG）")
                for q in similar_questions:
                    st.markdown(f"```\n{q}\n```")
    except Exception as e:
        st.warning(f"CSVファイルの読み込みに失敗しました。RAGなしで進めます。\n\n詳細: {e}")
        rag_text = ""

    # === GPTによる解説生成 =========================================
    with st.spinner("GPTが解説を生成中..."):
        prompt_text = (
            "以下の画像に含まれる問題に対して、正解とその根拠を説明し、各選択肢に対する解説をである調で記述せよ。"
            + (f"\n以下は過去問から抽出した類似問題情報である：\n{rag_text}" if rag_text else "")
        )

        response = client.chat.completions.create(
            model="gpt-4o-2024-11-20",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_uri}},
                        {"type": "text", "text": prompt_text}
                    ]
                }
            ],
            max_tokens=2000,
            temperature=0.1
        )

        result = response.choices[0].message.content.strip()
        st.subheader("💡 GPTの解説結果（構造化表示）")

        overview = ""
        answer = ""

        overview_match = re.search(r"【?問題の概要】?\n?(.*?)(?=\n【|$)", result, re.DOTALL)
        if overview_match:
            overview = overview_match.group(1).strip()

        answer_match = re.search(r"【?正解】?\n?(.*?)(?=\n【|$)", result, re.DOTALL)
        if answer_match:
            answer = answer_match.group(1).strip()

        if overview:
            st.markdown("### 📝 問題の概要")
            st.markdown(overview)

        if answer:
            st.markdown("### ✅ 正解")
            st.markdown(answer)
