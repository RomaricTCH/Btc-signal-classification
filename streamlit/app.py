import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

# =====================================================
# STYLE MATPLOTLIB
# =====================================================

plt.rcParams.update({
    "figure.figsize": (7, 3.5),
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8
})

# =====================================================
# CONFIG
# =====================================================

st.set_page_config(
    page_title="BTCUSD Trading Signal",
    layout="wide"
)

st.title("BTCUSD Trading Signal — BUY / HOLD / SELL")

# =====================================================
# PATHS
# =====================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "final" / "BTCUSD_final_dataset.csv"
MODEL_PATH = BASE_DIR / "models" / "gradient_boosting_final.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler_final.pkl"
CONFIG_PATH = BASE_DIR / "models" / "final_config.pkl"
FEATURES_PATH = BASE_DIR / "models" / "final_features.pkl"

# =====================================================
# LOAD
# =====================================================

df = pd.read_csv(DATA_PATH)

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
config = joblib.load(CONFIG_PATH)
final_features = joblib.load(FEATURES_PATH)

buy_threshold = config["buy_threshold"]
sell_threshold = config["sell_threshold"]

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("Sommaire")

pages = [
    "Accueil",
    "Exploration",
    "DataVisualization",
    "Modélisation",
    "Prédiction",
    "Conclusion"
]

page = st.sidebar.radio("Aller vers", pages)

st.sidebar.markdown("---")
st.sidebar.subheader("Configuration modèle")
st.sidebar.write("Modèle : Gradient Boosting")
st.sidebar.write(f"BUY threshold : {buy_threshold}")
st.sidebar.write(f"SELL threshold : {sell_threshold}")

# =====================================================
# PREDICTION
# =====================================================

X = df[final_features]

last_row = X.iloc[[-1]]

last_row_scaled = scaler.transform(last_row)

proba = model.predict_proba(last_row_scaled)[0]

classes = model.classes_

proba_dict = dict(zip(classes, proba))

sell_prob = proba_dict.get(-1, 0)
hold_prob = proba_dict.get(0, 0)
buy_prob = proba_dict.get(1, 0)

if buy_prob >= buy_threshold:
    signal = "BUY"

elif sell_prob >= sell_threshold:
    signal = "SELL"

else:
    signal = "HOLD"

# =====================================================
# ACCUEIL
# =====================================================

if page == "Accueil":

    st.header("Présentation du projet")

    st.markdown("""
    Ce projet vise à transformer des données historiques BTCUSD en signaux d'aide à la décision :

    - BUY
    - HOLD
    - SELL

    Le modèle est pensé pour une logique de swing trading court terme,
    avec un horizon de décision d'environ 5 jours.
    """)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Observations", df.shape[0])
    col2.metric("Features finales", len(final_features))
    col3.metric("Horizon", "5 jours")
    col4.metric("Signal actuel", signal)

    st.subheader("Pipeline du projet")

    st.markdown("""
    FMP → Cleaning → Feature Engineering → Target dynamique → EDA →
    Modélisation → GridSearch → Threshold tuning → Streamlit
    """)

# =====================================================
# EXPLORATION
# =====================================================

elif page == "Exploration":

    st.header("Exploration des données")

    n_rows = st.slider(
        "Nombre de lignes à afficher",
        min_value=5,
        max_value=50,
        value=10
    )

    st.dataframe(
        df.head(n_rows),
        use_container_width=True
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Lignes", df.shape[0])
    col2.metric("Colonnes", df.shape[1])
    col3.metric("Features finales", len(final_features))

    st.subheader("Distribution de la target")

    target_counts = (
        df["target"]
        .value_counts()
        .sort_index()
    )

    target_df = pd.DataFrame({
        "Classe": ["SELL", "HOLD", "BUY"],
        "Nombre": target_counts.values
    })

    col_graph, col_text = st.columns([2, 1])

    with col_graph:

        fig, ax = plt.subplots(figsize=(6, 3.5))

        ax.bar(
            target_df["Classe"],
            target_df["Nombre"]
        )

        ax.set_title("Distribution des classes")
        ax.set_ylabel("Nombre d'observations")
        ax.grid(axis="y", alpha=0.4)

        st.pyplot(fig)

    with col_text:

        st.write("""
        La target reste relativement équilibrée.

        Le problème de performance vient donc surtout de la difficulté
        du marché BTC plutôt que d'un fort déséquilibre des classes.
        """)

    if st.checkbox("Afficher les valeurs manquantes"):

        st.dataframe(
            df.isna().sum(),
            use_container_width=True
        )

# =====================================================
# DATAVISUALIZATION
# =====================================================

elif page == "DataVisualization":

    st.header("DataVisualization")

    graph_choice = st.selectbox(
        "Choisir une visualisation",
        [
            "Distribution des rendements",
            "Distribution de la volatilité",
            "Skewness / Kurtosis",
            "Importance familles"
        ]
    )

    col_graph, col_text = st.columns([2, 1])

    # =========================================
    # DISTRIBUTION RENDEMENTS
    # =========================================

    if graph_choice == "Distribution des rendements":

        with col_graph:

            fig, ax = plt.subplots(figsize=(7, 3.5))

            ax.hist(
                df["return_1"],
                bins=60
            )

            ax.set_title(
                "Distribution des rendements journaliers"
            )

            ax.set_xlabel("Return 1")
            ax.set_ylabel("Fréquence")
            ax.grid(True, alpha=0.4)

            st.pyplot(fig)

        with col_text:

            st.subheader("Lecture")

            st.write("""
            Les rendements sont fortement concentrés autour de zéro.

            Les queues épaisses indiquent que les mouvements extrêmes
            sont fréquents sur Bitcoin.
            """)

            st.info("""
            Les outliers sont conservés car ils représentent souvent
            de vrais événements de marché.
            """)

    # =========================================
    # VOLATILITE
    # =========================================

    elif graph_choice == "Distribution de la volatilité":

        with col_graph:

            fig, ax = plt.subplots(figsize=(7, 3.5))

            ax.hist(
                df["volatility_20"],
                bins=60
            )

            ax.set_title(
                "Distribution de la volatilité 20 jours"
            )

            ax.set_xlabel("Volatility 20")
            ax.set_ylabel("Fréquence")
            ax.grid(True, alpha=0.4)

            st.pyplot(fig)

        with col_text:

            st.subheader("Lecture")

            st.write("""
            Le marché alterne entre périodes calmes
            et périodes de forte instabilité.
            """)

            st.warning("""
            La volatilité devient donc centrale
            dans la construction de la target.
            """)

    # =========================================
    # SKEWNESS / KURTOSIS
    # =========================================

    elif graph_choice == "Skewness / Kurtosis":

        selected_vars = [
            "return_1",
            "return_3",
            "return_7",
            "volatility_5",
            "volatility_20",
            "rsi",
            "BB_width"
        ]

        stats = []

        for col in selected_vars:

            if col in df.columns:

                stats.append({
                    "Variable": col,
                    "Skewness": round(df[col].skew(), 3),
                    "Kurtosis": round(df[col].kurt(), 3)
                })

        stats_df = pd.DataFrame(stats)

        with col_graph:

            st.dataframe(
                stats_df,
                use_container_width=True
            )

        with col_text:

            st.subheader("Lecture")

            st.write("""
            Les fortes valeurs de kurtosis confirment
            des queues épaisses et des mouvements extrêmes.
            """)

    # =========================================
    # FAMILLES
    # =========================================

    elif graph_choice == "Importance familles":

        family_df = pd.DataFrame({

            "Famille": [
                "Bandes Bollinger",
                "Volatilité",
                "Momentum / Rendement",
                "Tendance",
                "Volume"
            ],

            "Importance": [
                42.2,
                24.9,
                13.5,
                10.9,
                8.5
            ]
        })

        with col_graph:

            fig, ax = plt.subplots(figsize=(5, 5))

            ax.pie(
                family_df["Importance"],
                labels=family_df["Famille"],
                autopct="%1.1f%%",
                textprops={"fontsize": 8}
            )

            ax.set_title(
                "Répartition des familles d'indicateurs"
            )

            st.pyplot(fig)

        with col_text:

            st.subheader("Lecture")

            st.write("""
            Le modèle s'appuie surtout sur :

            - les Bandes de Bollinger
            - la volatilité

            Il analyse principalement le contexte du marché.
            """)

# =====================================================
# MODELISATION
# =====================================================

elif page == "Modélisation":

    st.header("Modélisation")

    st.markdown("""
    Trois modèles ont été comparés :

    - Logistic Regression
    - RandomForest
    - GradientBoosting

    Le modèle retenu est GradientBoosting car il présente
    le meilleur compromis global.
    """)

    # =========================================
    # RESULTATS MODELES
    # =========================================

    st.subheader("Comparaison des modèles")

    model_results = pd.DataFrame({

        "Modèle": [
            "GradientBoosting",
            "RandomForest",
            "Logistic Regression"
        ],

        "Accuracy": [
            0.345,
            0.338,
            0.259
        ],

        "F1 BUY": [
            0.296,
            0.187,
            0.045
        ],

        "F1 SELL": [
            0.318,
            0.413,
            0.411
        ],

        "F1 Macro": [
            0.341,
            0.312,
            0.152
        ]
    })

    st.dataframe(
        model_results,
        use_container_width=True
    )

    metric = st.selectbox(
        "Choisir une métrique",
        ["Accuracy", "F1 BUY", "F1 SELL", "F1 Macro"]
    )

    fig, ax = plt.subplots(figsize=(7, 3.5))

    ax.bar(
        model_results["Modèle"],
        model_results[metric]
    )

    ax.set_title(
        f"Comparaison des modèles — {metric}"
    )

    ax.set_ylabel(metric)

    ax.grid(axis="y", alpha=0.4)

    st.pyplot(fig)

    # =========================================
    # THRESHOLD TUNING
    # =========================================

    st.subheader("Threshold tuning")

    threshold_df = pd.DataFrame({

        "Seuil": [0.30, 0.35, 0.40, 0.45, 0.50],

        "F1 BUY": [
            0.347,
            0.278,
            0.167,
            0.100,
            0.088
        ],

        "F1 SELL": [
            0.318,
            0.355,
            0.337,
            0.274,
            0.213
        ],

        "F1 Macro": [
            0.299,
            0.352,
            0.335,
            0.305,
            0.292
        ],

        "Accuracy": [
            0.305,
            0.359,
            0.378,
            0.390,
            0.411
        ]
    })

    st.dataframe(
        threshold_df,
        use_container_width=True
    )

    threshold_metric = st.selectbox(
        "Métrique threshold tuning",
        ["F1 BUY", "F1 SELL", "F1 Macro", "Accuracy"]
    )

    fig, ax = plt.subplots(figsize=(7, 3.5))

    ax.plot(
        threshold_df["Seuil"],
        threshold_df[threshold_metric],
        marker="o"
    )

    ax.set_title(
        f"Impact du seuil sur {threshold_metric}"
    )

    ax.set_xlabel("Seuil")
    ax.set_ylabel(threshold_metric)

    ax.grid(True, alpha=0.4)

    st.pyplot(fig)

    st.success("Seuil final retenu : 0.35")

    # =========================================
    # TEST ROBUSTESSE
    # =========================================

    st.subheader("Test de robustesse sur période récente")

    st.markdown("""
    Le dataset court 2021–2025 est inclus dans l'historique global BTCUSD.

    Il ne s'agit donc pas d'une validation externe totalement indépendante.

    Il est utilisé ici comme test de robustesse afin d'observer
    comment le modèle se comporte sur une période plus récente du marché.
    """)

    robustness_df = pd.DataFrame({

        "Dataset": [
            "BTC long (test only)",
            "BTC court 2021-2025"
        ],

        "Accuracy": [
            0.358794,
            0.444578
        ],

        "F1 BUY": [
            0.278481,
            0.325856
        ],

        "F1 SELL": [
            0.354793,
            0.407787
        ],

        "F1 Macro": [
            0.352013,
            0.423127
        ]
    })

    st.dataframe(
        robustness_df,
        use_container_width=True
    )

    robustness_metric = st.selectbox(
        "Métrique robustesse",
        ["Accuracy", "F1 BUY", "F1 SELL", "F1 Macro"]
    )

    fig, ax = plt.subplots(figsize=(7, 3.5))

    ax.bar(
        robustness_df["Dataset"],
        robustness_df[robustness_metric]
    )

    ax.set_title(
        f"Test de robustesse — {robustness_metric}"
    )

    ax.set_ylabel(
        robustness_metric
    )

    ax.grid(axis="y", alpha=0.4)

    st.pyplot(fig)

    st.info("""
    Le modèle conserve une certaine capacité à générer des signaux
    sur une période récente, mais ce test ne remplace pas
    une vraie validation externe indépendante.
    """)

# =====================================================
# PREDICTION
# =====================================================

elif page == "Prédiction":

    st.header("Prédiction finale")

    if signal == "BUY":

        st.success("Signal actuel : BUY")

    elif signal == "SELL":

        st.error("Signal actuel : SELL")

    else:

        st.warning("Signal actuel : HOLD")

    col1, col2, col3 = st.columns(3)

    col1.metric("SELL", round(sell_prob, 3))
    col2.metric("HOLD", round(hold_prob, 3))
    col3.metric("BUY", round(buy_prob, 3))

    st.subheader("Probabilités du modèle")

    proba_df = pd.DataFrame({

        "Classe": ["SELL", "HOLD", "BUY"],

        "Probabilité": [
            sell_prob,
            hold_prob,
            buy_prob
        ]
    })

    st.bar_chart(
        proba_df.set_index("Classe")
    )

    if st.checkbox(
        "Afficher la dernière ligne utilisée"
    ):

        st.dataframe(
            last_row,
            use_container_width=True
        )

# =====================================================
# CONCLUSION
# =====================================================

elif page == "Conclusion":

    st.header("Conclusion opérationnelle")

    st.markdown("""
    Le modèle constitue une aide à la décision orientée
    vers une logique de swing trading court terme.
    """)

    st.subheader("Ce que regarde le modèle")

    st.markdown("""
    Le modèle regarde principalement :

    - l'état du marché
    - la volatilité
    - la position du prix
    - l'intensité des mouvements récents

    Il cherche donc surtout à reconnaître
    des contextes historiquement favorables ou défavorables.
    """)

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Forces")

        st.markdown("""
        - Pipeline complet
        - EDA approfondie
        - Threshold tuning
        - Interprétation des variables
        - Streamlit interactif
        """)

    with col2:

        st.subheader("Limites")

        st.markdown("""
        - Performances modestes
        - Pas encore de backtest complet
        - Une seule crypto principale
        - Sensibilité aux régimes de marché
        """)

    st.subheader("Perspectives")

    st.markdown("""
    - Backtesting complet
    - Test sur ETH
    - Clustering des régimes de marché
    - Données macro et sentiment
    """)