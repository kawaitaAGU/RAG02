import streamlit as st
import base64
import io
import pandas as pd
from PIL import Image
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# === OpenAI APIキーの初期化（Secrets） ============================
if "OPENAI_API_KEY" not in st.secrets:
    st.error("OPENAI_API_KEY が未設定です。StreamlitのSecretsに追加してください。")
    st.stop()

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# === UIタイトルとファイルアップロード ===============================
st.title("画像から問題を読み取り、RAG付きで自動解説")

uploaded_img = st.file_uploader("問題画像をアップロード（.png, .jpg）", type=["png", "jpg", "jpeg"])
uploaded_excel = st.file_uploader("過去問Excelファイルをアップロード（.xlsx、任意）", type=["xlsx"])

if uploaded_img:
    # === 画像のbase64データをセッションに保存 =======================
    if 'b64_img' not in st.session_state:
        image = Image.open(uploaded_img).convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        st.session_state['b64_img'] = base64.b64encode(buffer.getvalue()).decode()

    data_uri = f"data:image/png;base64,{st.session_state['b64_img']}"

    # === GPTでOCR（画像から問題文を抽出） ===========================
    with st.spinner("画像から問題文を抽出中..."):
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

    # === ExcelからRAG抽出（あれば） ==================================
    rag_text = ""
    if uploaded_excel is not None:
        try:
            with st.spinner("Excelから類似問題を検索中..."):
                df = pd.read_excel(uploaded_excel)
                corpus, index_to_row = [], []

                for i, row in df.iterrows():
                    for cell in row:
                        if isinstance(cell, str) and len(cell) > 10:
                            corpus.append(cell)
                            index_to_row.append(i)

                if corpus:
                    vectorizer = TfidfVectorizer()
                    X = vectorizer.fit_transform(corpus + [query_text])
                    similarities = cosine_similarity(X[-1], X[:-1])[0]
                    top_indices = similarities.argsort()[-3:][::-1]

                    similar_questions = []
                    for idx in top_indices:
                        row = df.iloc[index_to_row[idx]]
                        text = corpus[idx]
                        choices = [str(cell) for cell in row if isinstance(cell, str) and 5 < len(cell) < 100 and cell != text]
                        correct = next((cell.strip().upper() for cell in row if isinstance(cell, str) and cell.strip().upper() in ['A', 'B', 'C', 'D', 'E']), "")

                        qinfo = f"{text}\n選択肢:\n" + "\n".join(f"- {c}" for c in choices[:5])
                        if correct:
                            qinfo += f"\n正解と思われる選択肢: {correct}"
                        similar_questions.append(qinfo)

                    rag_text = "\n\n".join(similar_questions)
                    st.subheader("📚 類似問題（RAG）")
                    for q in similar_questions:
                        st.markdown(f"```\n{q}\n```")
        except Exception as e:
            st.warning(f"Excelファイルの読み込みに失敗しました。RAGなしで進めます。\n\n詳細: {e}")
            rag_text = ""

    # === GPTに解説を依頼（自動実行） ================================
    with st.spinner("GPTが問題の解釈と解説を生成中..."):
        prompt_text = (
            f"今送った画像の問題の解説をしてください。正解を明示し、根拠を説明してください。"
            + (f"\n以下は過去問から抽出した類似問題情報です：\n{rag_text}" if rag_text else "")
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

        # === 結果を構造化して表示 ====================================
        overview = ""
        answer = ""
        choices = {}

        overview_match = re.search(r"【?問題の概要】?\n?(.*?)(?=\n【|$)", result, re.DOTALL)
        if overview_match:
            overview = overview_match.group(1).strip()

        answer_match = re.search(r"【?正解】?\n?(.*?)(?=\n【|$)", result, re.DOTALL)
        if answer_match:
            answer = answer_match.group(1).strip()

        choice_matches = re.findall(
            r"^([①-⑤1-5a-eA-Eａ-ｅＡ-Ｅ])[:：]?\s*(.+?)(?=\n[①-⑤1-5a-eA-Eａ-ｅＡ-Ｅ][:：]|\n*$)",
            result, re.MULTILINE | re.DOTALL
        )
        for label, text in choice_matches:
            choices[label.strip()] = text.strip()

        if overview:
            st.markdown("### 📝 問題の概要")
            st.markdown(overview)

        if answer:
            st.markdown("### ✅ 正解")
            st.markdown(answer)

        if choices:
            st.markdown("### 🔍 選択肢の解説")
            for label, text in choices.items():
                st.markdown(f"**{label}**: {text}")
        else:
            st.markdown("### 📄 解説（分割できなかった場合）")
            st.markdown(result)
