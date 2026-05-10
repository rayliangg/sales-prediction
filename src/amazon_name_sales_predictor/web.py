from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

try:
    from .predict import predict_one
except ImportError:
    # Support `streamlit run src/.../web.py` script execution.
    from amazon_name_sales_predictor.predict import predict_one


def main() -> None:
    st.set_page_config(page_title="Product Name Sales Predictor", page_icon="🛒", layout="wide")
    st.title("Amazon Product Naming vs Sales Predictor")
    st.caption("輸入產品名稱，預測可能銷售量，並顯示在相似商品中的正規化表現。")

    with st.sidebar:
        st.header("設定")
        model_path_str = st.text_input("模型路徑", value="models/name_sales_model.joblib")
        top_k = st.slider("相似商品數量 (Top K)", min_value=10, max_value=100, value=20, step=5)

    model_path = Path(model_path_str)
    if not model_path.exists():
        st.warning("找不到模型檔，請先訓練模型。")
        st.code("PYTHONPATH=src python -m amazon_name_sales_predictor.train --output-dir models")
        return

    product_name = st.text_input(
        "產品名稱",
        value="Wireless Bluetooth Earbuds Noise Cancelling",
        help="輸入你想評估的產品命名。",
    )

    col_a, col_b = st.columns([1, 3])
    with col_a:
        run_predict = st.button("開始預測", type="primary", use_container_width=True)
    with col_b:
        st.caption("建議：命名中包含功能詞、規格詞與類目詞，通常更容易對應到正確需求。")

    if run_predict:
        with st.spinner("模型推論中..."):
            try:
                result = predict_one(model_path=model_path, product_name=product_name, top_k=top_k)
            except Exception as exc:  # pragma: no cover
                st.error(f"預測失敗：{exc}")
                return

        st.subheader("預測結果")
        kpi_1, kpi_2, kpi_3, kpi_4 = st.columns(4)
        kpi_1.metric("預測銷售量", f"{result['predicted_sales']:.2f}")
        kpi_2.metric("同質平均銷量", f"{result['similar_products_avg_sales']:.2f}")
        kpi_3.metric("同質百分位", f"{result['normalized_percentile_in_similars']:.1f}%")
        kpi_4.metric("標準分數 Z", f"{result['normalized_z_score']:.2f}")

        st.write(f"推斷類別：`{result['inferred_category']}`")

        st.subheader(f"相似商品參考（Top {len(result['top_similar_examples'])}）")
        neighbors_df = pd.DataFrame(result["top_similar_examples"])
        st.dataframe(neighbors_df, use_container_width=True, hide_index=True)

        st.subheader("解讀建議")
        percentile = result["normalized_percentile_in_similars"]
        if percentile >= 75:
            st.success("你的命名在同類商品中偏強，具有不錯的銷量潛力。")
        elif percentile >= 40:
            st.info("你的命名在同類商品中屬於中段，可優化關鍵詞提升辨識度。")
        else:
            st.warning("你的命名在同類商品中偏弱，建議補強產品核心賣點與規格詞。")


if __name__ == "__main__":
    main()

