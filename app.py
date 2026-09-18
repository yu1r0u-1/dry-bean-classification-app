#1 ライブラリの設定
import streamlit as st
import joblib
from pathlib import Path
import pandas as pd
import streamlit as st


#2 ページの設定
st.set_page_config(
    page_title="Seed Sort AI",
    page_icon="🫘",
    layout="wide"
)

st.title("🫘 seed sort AI")
st.subheader("豆の形状データから品種を予測するアプリ")

st.write(
    """
    豆の面積や周囲長などの特徴量を入力すると、学習済みの機械学習モデルが豆の品種を予測します
    """
)

st.info("現在、予測機能を開発中です")

#3 モデルの設定
model_path=Path(__file__).parent/"models"/"bean_model.joblib"
model_data=joblib.load(model_path)
model=model_data["model"]

#4　入力フォーム
with st.form("prediction_form"):
    st.subheader("豆の特徴量を入力してください")
    left_column,right_column=st.columns(2)

    with left_column:
        area=st.number_input(
            "Area(面積)",
            min_value=0.0,
            value=53000.0
        )

        perimeter=st.number_input(
            "Perimeter(周囲長)",
            min_value=0.0,
            value=850.0
        )

        major_axis_length=st.number_input(
            "MajorAxilength(長軸の長さ)",
            min_value=0.0,
            value=320.0
        )

        minor_axis_length = st.number_input(
            "MinorAxisLength（短軸の長さ）",
            min_value=0.0,
            value=200.0
        )

        aspect_ration = st.number_input(
            "AspectRation（縦横比）",
            min_value=0.0,
            value=1.5
        )

        eccentricity = st.number_input(
            "Eccentricity（離心率）",
            min_value=0.0,
            max_value=1.0,
            value=0.75
        )

        convex_area = st.number_input(
            "ConvexArea（凸包面積）",
            min_value=0.0,
            value=54000.0
        )

        equiv_diameter = st.number_input(
            "EquivDiameter（等価直径）",
            min_value=0.0,
            value=250.0
        )

    with right_column:

        extent = st.number_input(
            "Extent（外接矩形に占める割合）",
            min_value=0.0,
            max_value=1.0,
            value=0.75
        )

        solidity = st.number_input(
            "Solidity（充実度）",
            min_value=0.0,
            max_value=1.0,
            value=0.98
        )

        roundness = st.number_input(
            "roundness（円形度）",
            min_value=0.0,
            max_value=1.0,
            value=0.87
        )

        compactness = st.number_input(
            "Compactness（コンパクト度）",
            min_value=0.0,
            max_value=1.0,
            value=0.80
        )

        shape_factor1 = st.number_input(
            "ShapeFactor1",
            min_value=0.0,
            value=0.0065,
            format="%.6f"
        )

        shape_factor2 = st.number_input(
            "ShapeFactor2",
            min_value=0.0,
            value=0.0017,
            format="%.6f"
        )

        shape_factor3 = st.number_input(
            "ShapeFactor3",
            min_value=0.0,
            value=0.64
        )

        shape_factor4 = st.number_input(
            "ShapeFactor4",
            min_value=0.0,
            max_value=1.0,
            value=0.99
        )

    predict_button=st.form_submit_button(
        "品種を予測する",
        type="primary"
    )

#5 予測処理
if predict_button:

    input_data=pd.DataFrame(
        [{
            "Area": area,
            "Perimeter": perimeter,
            "MajorAxisLength": major_axis_length,
            "MinorAxisLength": minor_axis_length,
            "AspectRation": aspect_ration,
            "Eccentricity": eccentricity,
            "ConvexArea": convex_area,
            "EquivDiameter": equiv_diameter,
            "Extent": extent,
            "Solidity": solidity,
            "roundness": roundness,
            "Compactness": compactness,
            "ShapeFactor1": shape_factor1,
            "ShapeFactor2": shape_factor2,
            "ShapeFactor3": shape_factor3,
            "ShapeFactor4": shape_factor4
        }]
    )

    prediction=model.predict(input_data)[0]
    probabilities=model.predict_proba(input_data)[0]

    probability_data=pd.DataFrame({
        "品種":model.classes_,
        "予測確率":probabilities,

    }).sort_values("予測確率",ascending=False)
    st.success(f"予測された品種:{prediction}")
    st.write(
        "それぞれの品種である確率"
    )
    st.dataframe(probability_data)
    st.bar_chart(
        probability_data.set_index("品種")
    )