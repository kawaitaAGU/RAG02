# gpt-4o-2024-11-20
import streamlit as st
import base64
import io
import pandas as pd
from PIL import Image
from openai import OpenAI
from pathlib import Path

# === OpenAI APIキーの初期化 ========================================
if "OPENAI_API_KEY" not in st.secrets:
    st.error("OPENAI_API_KEY が未設定です。StreamlitのSecretsに追加してください。")
    st.stop()

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# === UIタイトルと画像アップロード欄 ===============================
st.title("画像から問題を読み取り、解説と類似問題を生成")

uploaded_img = st.file_uploader("問題画像をアップロード（.png, .jpg, .jpeg）", type=["png", "jpg", "jpeg"])

if uploaded_img:
    st.session_state.clear()

    st.image(uploaded_img, caption="アップロードされた画像", use_column_width=True)

    image = Image.open(uploaded_img).convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    b64_img = base64.b64encode(buffer.getvalue()).decode()
    data_uri = f"data:image/png;base64,{b64_img}"

    # === 過去問データ（CSV）を読み込み =============================
    csv_path = Path("sample.csv")
    if not csv_path.exists():
        st.error("sample.csv が見つかりません。ファイルをアプリと同じフォルダに配置してください。")
        st.stop()

    df = pd.read_csv(csv_path)

    # === GPTに画像とCSVの内容を送信（1回で処理） =====================
    with st.spinner("GPTが問題を解析中..."):
        csv_text = df.to_csv(index=False)

        prompt = (
            "以下の画像には1つの選択式問題が含まれている。画像から問題文、選択肢、正解を抽出せよ。"
            "その後、各選択肢に対して説明を加え、最終的な正解を明示しなさい。"
            "また、以下に与えられた過去問のデータ（CSV形式）を参考にして、類似した問題を3問新たに作成し、"
            "それぞれに選択肢・正解・各選択肢の説明をつけて出力せよ。"
        )

        response = client.chat.completions.create(
            model="gpt-4o-2024-11-20",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_uri}},
                        {"type": "text", "text": prompt + "\n\n【過去問データ（CSV）】\n" + csv_text}
                    ]
                }
            ],
            max_tokens=4000,
            temperature=0.3
        )

        result = response.choices[0].message.content.strip()

    # === 出力表示（構造化なし、一括） ================================
    st.subheader("📘 GPTによる解析結果")
    st.markdown(result)
