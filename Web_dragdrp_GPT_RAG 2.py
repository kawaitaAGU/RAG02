#gpt-4o-2024-11-20
import streamlit as st
import base64
import io
import pandas as pd
from PIL import Image
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import re

# === OpenAI APIキーの初期化 ========================================
if "OPENAI_API_KEY" not in st.secrets:
    st.error("OPENAI_API_KEY が未設定です。StreamlitのSecretsに追加してください。")
    st.stop()

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# === UIタイトルと画像アップロード欄 ===============================
st.title("画像から問題を読み取り、RAG付きで自動解説＋類似問題生成")

uploaded_img = st.file_uploader("問題画像をアップロード（.png, .jpg）", type=["png", "jpg", "jpeg"])

if uploaded_img:
    st.session_state.clear()

    st.image(uploaded_img, caption="アップロードされた画像", use_column_width=True)

    image = Image.open(uploaded_img).convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    st.session_state['b64_img'] = base64.b64encode(buffer.getvalue()).decode()
    data_uri = f"data:image/png;base64,{st.session_state['b64_img']}"

    # === GPTによる画像処理（問題文抽出とRAG用データ生成） ===================
    with st.spinner("画像をGPTで解析中..."):
        extract_response = client.chat.completions.create(
            model="gpt-4o-2024-11-20",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_uri}},
                        {"type": "text", "text": "この画像に書かれている問題文、選択肢、正解、解説、各選択肢の説明を抽出してください。"}
                    ]
                }
            ],
            max_tokens=3000,
            temperature=0.0
        )
        extracted_text = extract_response.choices[0].message.content.strip()

    st.subheader("🔍 GPTによる問題情報の抽出")
    st.markdown(extracted_text)

    # === RAG処理用：CSVファイル読み込み ===============================
    rag_text = ""
    csv_path = Path("sample.csv")
    if not csv_path.exists():
        st.error("sample.csv が見つかりません。ファイルをアプリと同じフォルダに配置してください。")
        st.stop()

    try:
        with st.spinner("過去問データから類似問題を検索中..."):
            df = pd.read_csv(csv_path)
            if "問題文" not in df.columns:
                st.error("CSVファイルに '問題文' 列が必要です。")
                st.stop()

            corpus = df["問題文"].fillna("").tolist()
            vectorizer = TfidfVectorizer()
            X = vectorizer.fit_transform(corpus + [extracted_text])
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
    except Exception as e:
        st.warning(f"CSV読み込み失敗：{e}")

    # === GPTで解説・再評価・類似問題生成 =============================
    with st.spinner("GPTによる最終解説と類似問題の生成中..."):
        final_prompt = (
            "以下は画像から抽出された問題と解説、選択肢である。正解とその根拠を明示し、各選択肢に対する解説を述べよ。"
            "その後、同様の形式の類似問題を3問作成し、それぞれに選択肢と正解を示し、各選択肢についても説明せよ。"
            + f"\n\n問題情報:\n{extracted_text}"
            + (f"\n\n以下は参考となる過去問情報である：\n{rag_text}" if rag_text else "")
        )

        final_response = client.chat.completions.create(
            model="gpt-4o-2024-11-20",
            messages=[{"role": "user", "content": final_prompt}],
            max_tokens=4000,
            temperature=0.3
        )
        final_output = final_response.choices[0].message.content.strip()

    # === 結果の表示 ================================================
    st.subheader("✅ 最終解説と類似問題（GPT生成）")
    st.markdown(final_output)
