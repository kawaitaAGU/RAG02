# app.py
import streamlit as st
import base64
import io
import pandas as pd
from PIL import Image
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

# OpenAI API キーの確認
if "OPENAI_API_KEY" not in st.secrets:
    st.error("OPENAI_API_KEY が設定されていません。StreamlitのSecretsに設定してください。")
    st.stop()

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.title("🧠 画像から問題を読み取り、類似問題と解説を生成")

uploaded_img = st.file_uploader("🖼️ 問題画像をアップロード", type=["png", "jpg", "jpeg"])
csv_path = Path("sample.csv")

if not csv_path.exists():
    st.error("❌ sample.csv が見つかりません。アプリと同じディレクトリに配置してください。")
    st.stop()

if uploaded_img:
    st.image(uploaded_img, caption="アップロードされた画像", use_column_width=True)

    # Base64化してGPTに送信
    image = Image.open(uploaded_img).convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    b64_img = base64.b64encode(buffer.getvalue()).decode()
    data_uri = f"data:image/png;base64,{b64_img}"

    with st.spinner("🔍 問題文を抽出中..."):
        extract_response = client.chat.completions.create(
            model="gpt-4o-2024-11-20",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_uri}},
                        {"type": "text", "text": "あなたは歯学系の国家試験の専門家です。\n"
                "受験生に対して厳密かつ明確に、問題の正答とその理由を説明してください。\n"
                "文体は『である調』を使用し、選択肢ごとに簡潔な解説をつけてください" "この画像に含まれる問題文、選択肢、正解、各選択肢の解説を出力してください。である調で書いてください。"}
                    ]
                }
            ],
            max_tokens=2000,
            temperature=0.3
        )
        extracted = extract_response.choices[0].message.content.strip()

    st.subheader("📝 GPTによる問題解析")
    st.markdown(extracted)

    # 類似検索のためにcsv読み込み
    with st.spinner("📚 類似問題を検索中..."):
        df = pd.read_csv(csv_path)
        if "問題文" not in df.columns:
            st.error("❌ sample.csv に '問題文' 列が含まれていません。")
            st.stop()

        corpus = df["問題文"].fillna("").tolist()
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(corpus + [extracted])
        sims = cosine_similarity(X[-1], X[:-1])[0]
        top_idx = sims.argsort()[-3:][::-1]

        similar_qs = []
        for i in top_idx:
            row = df.iloc[i]
            q = row["問題文"]
            choices = [str(row.get(c, "")) for c in ['a', 'b', 'c', 'd', 'e']]
            answer = str(row.get("解答", ""))
            qtext = f"{q}\n選択肢:\n" + "\n".join([f"- {c}" for c in choices if c]) + f"\n正解: {answer}"
            similar_qs.append(qtext)

        rag_text = "\n\n".join(similar_qs)

    st.subheader("📖 類似問題（RAGによる検索）")
    for q in similar_qs:
        st.markdown(f"```\n{q}\n```")

    # GPTに再送信して、類似問題と解説を生成
    with st.spinner("🤖 GPTが類似問題と解説を生成中..."):
        second_prompt = (
            "次の内容は画像から抽出された問題と過去の類似問題である。\n\n"
            f"{extracted}\n\n"
            f"以下は類似する過去問データである：\n{rag_text}\n\n"
            "この内容を参考にして、問題の正解と解説をである調で再提示し、さらに類似した新しい問題を3問作成し、各選択肢についても簡潔な解説を加えてください。"
        )

        second_response = client.chat.completions.create(
            model="gpt-4o-2024-11-20",
            messages=[{"role": "user", "content": second_prompt}],
            max_tokens=2000,
            temperature=0.5
        )
        result = second_response.choices[0].message.content.strip()

    st.subheader("✅ GPTによる最終解説と類似問題の生成結果")
    st.markdown(result)
