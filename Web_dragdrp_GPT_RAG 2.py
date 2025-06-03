import re

# === GPT解説ボタン ================================================
if st.button("📘 GPTに解説を依頼する"):
    with st.spinner("GPTが問題の解釈と解説を生成中..."):
        prompt_text = (
            f"今送った画像の問題の解説をしてください。正解を明示し、根拠を説明してください。"
            f"{'以下は過去問から抽出した類似問題情報です：' + rag_text if rag_text else ''}"
        )

        response = client.chat.completions.create(
            model="gpt-4o",
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

        # === パース処理：概要・正解・選択肢別解説 ===================
        overview = ""
        answer = ""
        choices = {}

        # 概要抽出
        overview_match = re.search(r"【?問題の概要】?\n?(.*?)(?=\n【|$)", result, re.DOTALL)
        if overview_match:
            overview = overview_match.group(1).strip()

        # 正解抽出
        answer_match = re.search(r"【?正解】?\n?(.*?)(?=\n【|$)", result, re.DOTALL)
        if answer_match:
            answer = answer_match.group(1).strip()

        # 選択肢ごとの説明抽出
        choice_matches = re.findall(r"^([①-⑤1-5a-eA-Eａ-ｅＡ-Ｅ])[:：]?\s*(.+?)(?=\n[①-⑤1-5a-eA-Eａ-ｅＡ-Ｅ][:：]|\n*$)", result, re.MULTILINE | re.DOTALL)
        for label, text in choice_matches:
            choices[label.strip()] = text.strip()

        # === 表示処理 ============================================
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
