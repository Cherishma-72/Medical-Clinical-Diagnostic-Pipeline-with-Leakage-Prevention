"""
Heart Failure Survival Predictor
A leak-free clinical ML pipeline, shown live with Streamlit.
Run locally with:  streamlit run app.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from scipy import stats

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, cross_val_score, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report
)
from sklearn.inspection import permutation_importance

import plotly.graph_objects as go

RANDOM_STATE = 42
sns.set_style("whitegrid")

NUMERIC_FEATURES = ["age", "creatinine_phosphokinase", "ejection_fraction",
                     "platelets", "serum_creatinine", "serum_sodium", "time"]
CATEGORICAL_FEATURES = ["anaemia", "diabetes", "high_blood_pressure", "sex", "smoking"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

st.set_page_config(
    page_title="Heart Failure Survival Predictor",
    page_icon="🫀",
    layout="wide",
)

# ----------------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------------
@st.cache_data
def load_data(file):
    return pd.read_csv(file)


# ----------------------------------------------------------------------------------
# Pipeline builder
# ----------------------------------------------------------------------------------
def build_preprocessor(numeric_features=NUMERIC_FEATURES, categorical_features=CATEGORICAL_FEATURES):
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(drop="if_binary", handle_unknown="ignore"))
    ])
    return ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])


def build_pipeline(classifier, numeric_features=NUMERIC_FEATURES, categorical_features=CATEGORICAL_FEATURES):
    return Pipeline(steps=[
        ("preprocessor", build_preprocessor(numeric_features, categorical_features)),
        ("classifier", classifier),
    ])


# ----------------------------------------------------------------------------------
# All heavy computation lives inside ONE cached function so Streamlit only
# recomputes it once per dataset, not on every widget interaction.
# ----------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def run_full_pipeline(df_raw: pd.DataFrame):
    df = df_raw.copy()
    X = df[ALL_FEATURES]
    y = df["DEATH_EVENT"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {"accuracy": "accuracy", "precision": "precision", "recall": "recall",
               "f1": "f1", "roc_auc": "roc_auc"}

    models = {
        "Logistic Regression": build_pipeline(
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
        "K-Nearest Neighbours": build_pipeline(KNeighborsClassifier(n_neighbors=7)),
        "SVM (RBF Kernel)": build_pipeline(
            SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE)),
        "Random Forest": build_pipeline(
            RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE)),
        "Gradient Boosting": build_pipeline(GradientBoostingClassifier(random_state=RANDOM_STATE)),
    }

    cv_results, fold_auc = {}, {}
    for name, pipe in models.items():
        res = cross_validate(pipe, X_train, y_train, cv=cv_strategy, scoring=scoring)
        cv_results[name] = {m: res[f"test_{m}"].mean() for m in scoring}
        fold_auc[name] = res["test_roc_auc"]

    summary_df = pd.DataFrame(cv_results).T.sort_values("roc_auc", ascending=False)
    best_name, second_name = summary_df.index[0], summary_df.index[1]
    t_stat, p_value = stats.ttest_rel(fold_auc[best_name], fold_auc[second_name])

    # Hyperparameter tuning on Random Forest
    param_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [None, 5, 10],
        "classifier__min_samples_leaf": [1, 2],
    }
    grid = GridSearchCV(
        estimator=build_pipeline(RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE)),
        param_grid=param_grid, cv=cv_strategy, scoring="roc_auc", n_jobs=-1
    )
    grid.fit(X_train, y_train)
    best_pipeline = grid.best_estimator_

    y_pred = best_pipeline.predict(X_test)
    y_proba = best_pipeline.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_proba)

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    prec, rec, _ = precision_recall_curve(y_test, y_proba)
    avg_prec = average_precision_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Survived", "Died"], output_dict=True)

    # Bootstrap CI for AUC
    rng = np.random.RandomState(RANDOM_STATE)
    y_test_arr, boot_aucs = y_test.values, []
    for _ in range(500):
        idx = rng.randint(0, len(y_test_arr), len(y_test_arr))
        if len(np.unique(y_test_arr[idx])) < 2:
            continue
        boot_aucs.append(roc_auc_score(y_test_arr[idx], y_proba[idx]))
    ci_lower, ci_upper = np.percentile(boot_aucs, [2.5, 97.5])

    # Permutation importance
    perm = permutation_importance(best_pipeline, X_test, y_test, scoring="roc_auc",
                                   n_repeats=15, random_state=RANDOM_STATE, n_jobs=-1)
    perm_df = pd.DataFrame({
        "feature": X_test.columns, "importance": perm.importances_mean, "std": perm.importances_std
    }).sort_values("importance", ascending=False)

    # ---- Leakage demonstration: scale-before-split vs correct pipeline ----
    demo_numeric = df[NUMERIC_FEATURES].fillna(df[NUMERIC_FEATURES].median())
    demo_y = df["DEATH_EVENT"]

    correct_pipe = Pipeline([("scaler", StandardScaler()),
                              ("clf", LogisticRegression(max_iter=1000, class_weight="balanced",
                                                          random_state=RANDOM_STATE))])
    correct_scores = cross_validate(correct_pipe, demo_numeric, demo_y, cv=cv_strategy, scoring=scoring)

    leaked_X = StandardScaler().fit_transform(demo_numeric)
    leaky_clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
    leaky_scores = cross_validate(leaky_clf, leaked_X, demo_y, cv=cv_strategy, scoring=scoring)

    leak_comparison = pd.DataFrame({
        "Correct (no leakage)": {m: correct_scores[f"test_{m}"].mean() for m in scoring},
        "Leaky (scaled before split)": {m: leaky_scores[f"test_{m}"].mean() for m in scoring},
    })

    # ---- WITH vs WITHOUT 'time' feature (temporal leakage) ----
    best_params = {k.replace("classifier__", ""): v for k, v in grid.best_params_.items()}

    with_time_pipe = build_pipeline(RandomForestClassifier(class_weight="balanced",
                                                             random_state=RANDOM_STATE, **best_params))
    with_time_scores = cross_val_score(with_time_pipe, X_train, y_train, cv=cv_strategy, scoring="roc_auc")

    numeric_no_time = [f for f in NUMERIC_FEATURES if f != "time"]
    without_time_pipe = build_pipeline(
        RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, **best_params),
        numeric_features=numeric_no_time
    )
    X_train_no_time = X_train.drop(columns=["time"])
    without_time_scores = cross_val_score(without_time_pipe, X_train_no_time, y_train,
                                           cv=cv_strategy, scoring="roc_auc")

    return {
        "df": df, "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
        "summary_df": summary_df, "best_name": best_name, "second_name": second_name,
        "p_value": p_value, "best_pipeline": best_pipeline, "best_params": grid.best_params_,
        "y_pred": y_pred, "y_proba": y_proba, "test_auc": test_auc,
        "fpr": fpr, "tpr": tpr, "prec": prec, "rec": rec, "avg_prec": avg_prec,
        "cm": cm, "report": report, "ci_lower": ci_lower, "ci_upper": ci_upper,
        "boot_aucs": boot_aucs, "perm_df": perm_df, "leak_comparison": leak_comparison,
        "with_time_scores": with_time_scores, "without_time_scores": without_time_scores,
    }


# ----------------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------------
st.sidebar.title("🫀 Heart Failure Predictor")
st.sidebar.markdown(
    "A **leak-free** clinical ML pipeline predicting patient survival after heart failure, "
    "built with Scikit-Learn. [Project #21 — B.Tech Final Year]"
)

uploaded_file = st.sidebar.file_uploader("Upload dataset CSV (optional)", type="csv")
default_path = "heart_failure_clinical_records_dataset.csv"

if uploaded_file is not None:
    df_raw = load_data(uploaded_file)
else:
    try:
        df_raw = load_data(default_path)
        st.sidebar.success(f"Loaded bundled dataset: {default_path}")
    except FileNotFoundError:
        st.sidebar.warning("Please upload heart_failure_clinical_records_dataset.csv to continue.")
        st.stop()

st.sidebar.metric("Total Patients", len(df_raw))
st.sidebar.metric("Death Rate", f"{df_raw['DEATH_EVENT'].mean() * 100:.1f}%")

with st.spinner("Training and validating models (only runs once per dataset)..."):
    artifacts = run_full_pipeline(df_raw)

# ----------------------------------------------------------------------------------
# Main title
# ----------------------------------------------------------------------------------
st.title("🫀 Medical Clinical Diagnostic Pipeline with Leakage Prevention")
st.caption("Live demonstration of a leak-free Scikit-Learn pipeline for heart failure survival prediction")

tab_overview, tab_eda, tab_models, tab_final, tab_leak, tab_importance, tab_predict = st.tabs([
    "🏠 Overview", "📊 EDA", "🤖 Model Comparison", "🎯 Final Model",
    "🔍 Leakage Demo", "🧬 Feature Importance", "🩺 Live Predictor"
])

# ----------------------------------------------------------------------------------
# TAB 1: Overview
# ----------------------------------------------------------------------------------
with tab_overview:
    st.header("Project Overview")
    st.markdown("""
This dashboard showcases a **leakage-free machine learning pipeline** built to predict whether a
patient survives after heart failure, using the UCI **Heart Failure Clinical Records Dataset**
(299 patients, 12 clinical features).

**What makes this pipeline leak-free?**
- All imputation, encoding, and scaling steps are fit **only on training data**, inside a
  Scikit-Learn `Pipeline` + `ColumnTransformer`.
- Cross-validation refits preprocessing fresh on every fold — no fold ever sees another fold's data.
- The held-out test set is touched exactly once, for final scoring.
- We even go a step further and demonstrate a *feature-level* leakage risk (the `time` variable)
  that structural pipeline discipline alone cannot catch — see the **Feature Importance** tab.
""")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Patients", len(df_raw))
    c2.metric("Survived", int((df_raw["DEATH_EVENT"] == 0).sum()))
    c3.metric("Died", int((df_raw["DEATH_EVENT"] == 1).sum()))
    c4.metric("Features", len(ALL_FEATURES))

    st.subheader("Sample of the Dataset")
    st.dataframe(df_raw.head(10), use_container_width=True)

# ----------------------------------------------------------------------------------
# TAB 2: EDA
# ----------------------------------------------------------------------------------
with tab_eda:
    st.header("Exploratory Data Analysis")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Target Distribution")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.countplot(x="DEATH_EVENT", data=df_raw, hue="DEATH_EVENT",
                       palette=["#2E86AB", "#E63946"], legend=False, ax=ax)
        ax.set_xlabel("0 = Survived | 1 = Died")
        st.pyplot(fig)

    with col2:
        st.subheader("Correlation with Target")
        corr_target = df_raw.corr()["DEATH_EVENT"].drop("DEATH_EVENT").sort_values()
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        colors = ["#E63946" if v > 0 else "#2E86AB" for v in corr_target.values]
        ax2.barh(corr_target.index, corr_target.values, color=colors)
        ax2.set_xlabel("Correlation with DEATH_EVENT")
        st.pyplot(fig2)

    st.subheader("Full Correlation Heatmap")
    fig3, ax3 = plt.subplots(figsize=(11, 8))
    sns.heatmap(df_raw.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax3)
    st.pyplot(fig3)

    st.subheader("Feature Distribution by Outcome")
    feature_choice = st.selectbox("Choose a clinical feature to explore:", NUMERIC_FEATURES, index=4)
    fig4, ax4 = plt.subplots(figsize=(7, 4))
    sns.boxplot(x="DEATH_EVENT", y=feature_choice, data=df_raw, hue="DEATH_EVENT",
                palette=["#2E86AB", "#E63946"], legend=False, ax=ax4)
    ax4.set_xlabel("0 = Survived | 1 = Died")
    st.pyplot(fig4)

# ----------------------------------------------------------------------------------
# TAB 3: Model Comparison
# ----------------------------------------------------------------------------------
with tab_models:
    st.header("Model Benchmarking — 5 Algorithms, Same Leak-Free Pipeline")
    st.markdown(
        "Each model below is wrapped in the **identical** preprocessing pipeline and evaluated with "
        "**5-fold Stratified Cross-Validation** on the training set only."
    )
    st.dataframe(artifacts["summary_df"].style.format("{:.4f}").highlight_max(axis=0, color="#c9f7c9"),
                 use_container_width=True)

    fig5, ax5 = plt.subplots(figsize=(8, 4))
    artifacts["summary_df"]["roc_auc"].sort_values().plot(kind="barh", color="#2E86AB", ax=ax5)
    ax5.set_xlabel("Mean CV ROC-AUC")
    ax5.set_xlim(0.5, 1.0)
    st.pyplot(fig5)

    best_name, second_name, p_value = artifacts["best_name"], artifacts["second_name"], artifacts["p_value"]
    st.subheader("Statistical Significance Test")
    st.write(
        f"Comparing top model **{best_name}** vs runner-up **{second_name}** using a paired t-test "
        f"on per-fold ROC-AUC scores:"
    )
    st.metric("p-value", f"{p_value:.4f}")
    if p_value < 0.05:
        st.success(f"p < 0.05 — the difference IS statistically significant.")
    else:
        st.info(
            f"p ≥ 0.05 — NOT statistically significant. With only 5 folds we cannot confidently claim "
            f"'{best_name}' truly beats '{second_name}'; either is a defensible choice."
        )

# ----------------------------------------------------------------------------------
# TAB 4: Final Model Performance
# ----------------------------------------------------------------------------------
with tab_final:
    st.header("Final Tuned Model — Random Forest")
    st.write("Best hyperparameters found via `GridSearchCV`:")
    st.json(artifacts["best_params"])

    c1, c2, c3 = st.columns(3)
    c1.metric("Test ROC-AUC", f"{artifacts['test_auc']:.4f}")
    c2.metric("95% CI Lower", f"{artifacts['ci_lower']:.4f}")
    c3.metric("95% CI Upper", f"{artifacts['ci_upper']:.4f}")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Confusion Matrix")
        fig6, ax6 = plt.subplots(figsize=(5, 4))
        sns.heatmap(artifacts["cm"], annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Survived", "Died"], yticklabels=["Survived", "Died"], ax=ax6)
        ax6.set_xlabel("Predicted"); ax6.set_ylabel("Actual")
        st.pyplot(fig6)

    with col2:
        st.subheader("ROC & Precision-Recall Curves")
        fig7, (axa, axb) = plt.subplots(1, 2, figsize=(10, 4))
        axa.plot(artifacts["fpr"], artifacts["tpr"], color="#E63946",
                 label=f"AUC={artifacts['test_auc']:.3f}")
        axa.plot([0, 1], [0, 1], "--", color="grey")
        axa.set_title("ROC Curve"); axa.legend()
        axb.plot(artifacts["rec"], artifacts["prec"], color="#2E86AB",
                 label=f"AP={artifacts['avg_prec']:.3f}")
        axb.set_title("Precision-Recall Curve"); axb.legend()
        st.pyplot(fig7)

    st.subheader("Classification Report")
    st.dataframe(pd.DataFrame(artifacts["report"]).T.round(3), use_container_width=True)

    st.subheader("Bootstrap Distribution of Test ROC-AUC (500 resamples)")
    fig8, ax8 = plt.subplots(figsize=(8, 4))
    sns.histplot(artifacts["boot_aucs"], bins=30, kde=True, color="#2E86AB", ax=ax8)
    ax8.axvline(artifacts["test_auc"], color="black", linestyle="--", label="Point estimate")
    ax8.legend()
    st.pyplot(fig8)

# ----------------------------------------------------------------------------------
# TAB 5: Leakage demo
# ----------------------------------------------------------------------------------
with tab_leak:
    st.header("🔍 Live Demonstration: Leakage Done Wrong")
    st.markdown("""
We deliberately compare two approaches:
- ✅ **Correct** — the scaler is fit fresh inside every cross-validation fold.
- ❌ **Leaky** — the scaler is fit once on the *entire* dataset before cross-validation, so
  validation-fold statistics leak into training.
""")
    leak_df = artifacts["leak_comparison"]
    st.dataframe(leak_df.style.format("{:.4f}"), use_container_width=True)

    fig9, ax9 = plt.subplots(figsize=(8, 4))
    leak_df.T.plot(kind="bar", ax=ax9, color=None)
    ax9.set_ylabel("Score"); ax9.set_ylim(0, 1)
    plt.xticks(rotation=0)
    st.pyplot(fig9)

    st.info(
        "The gap here is often small for a gentle transform like scaling — which is exactly why "
        "leakage is dangerous: it's easy to underestimate. In real pipelines, leakier steps "
        "(feature selection, target-aware imputation) can inflate accuracy by 10-30+ points."
    )

# ----------------------------------------------------------------------------------
# TAB 6: Feature importance
# ----------------------------------------------------------------------------------
with tab_importance:
    st.header("🧬 Permutation Feature Importance")
    fig10, ax10 = plt.subplots(figsize=(8, 5))
    pdf = artifacts["perm_df"]
    ax10.barh(pdf["feature"], pdf["importance"], xerr=pdf["std"], color="#2E86AB")
    ax10.invert_yaxis()
    ax10.set_xlabel("Mean decrease in ROC-AUC when shuffled")
    st.pyplot(fig10)

    st.header("⏱️ Temporal Leakage: The `time` Feature")
    st.markdown("""
`time` (follow-up duration) is one of the strongest predictors, but it would **not be known**
at the moment a real prediction is needed (a doctor cannot know in advance how long a patient
will be followed). We compare model performance **with** and **without** this feature.
""")
    wt, wot = artifacts["with_time_scores"], artifacts["without_time_scores"]
    c1, c2, c3 = st.columns(3)
    c1.metric("WITH time — mean AUC", f"{wt.mean():.4f}")
    c2.metric("WITHOUT time — mean AUC", f"{wot.mean():.4f}")
    c3.metric("Performance Drop", f"{wt.mean() - wot.mean():.4f}")

    fig11, ax11 = plt.subplots(figsize=(6, 4))
    ax11.boxplot([wt, wot], tick_labels=["With 'time'", "Without 'time'"])
    ax11.set_ylabel("ROC-AUC (5-fold CV)")
    st.pyplot(fig11)

# ----------------------------------------------------------------------------------
# TAB 7: Live Predictor
# ----------------------------------------------------------------------------------
with tab_predict:
    st.header("🩺 Live Patient Risk Predictor")
    st.markdown("Enter a patient's clinical values below to get a live survival risk prediction "
                "from the final tuned pipeline.")

    with st.form("patient_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age", min_value=18, max_value=110, value=60)
            ejection_fraction = st.slider("Ejection Fraction (%)", 10, 80, 38)
            serum_creatinine = st.number_input("Serum Creatinine (mg/dL)", 0.1, 10.0, 1.1, step=0.1)
            platelets = st.number_input("Platelets (kiloplatelets/mL)", 25000, 900000, 260000, step=1000)
        with c2:
            serum_sodium = st.slider("Serum Sodium (mEq/L)", 110, 150, 137)
            creatinine_phosphokinase = st.number_input("CPK Enzyme Level (mcg/L)", 20, 8000, 250)
            time_val = st.number_input("Follow-up Time (days)", 1, 300, 120)
            sex = st.radio("Sex", ["Female", "Male"], horizontal=True)
        with c3:
            anaemia = st.checkbox("Anaemia")
            diabetes = st.checkbox("Diabetes")
            high_blood_pressure = st.checkbox("High Blood Pressure")
            smoking = st.checkbox("Smoking")

        submitted = st.form_submit_button("🔮 Predict Survival")

    if submitted:
        patient = pd.DataFrame([{
            "age": age, "creatinine_phosphokinase": creatinine_phosphokinase,
            "ejection_fraction": ejection_fraction, "platelets": platelets,
            "serum_creatinine": serum_creatinine, "serum_sodium": serum_sodium,
            "time": time_val,
            "anaemia": int(anaemia), "diabetes": int(diabetes),
            "high_blood_pressure": int(high_blood_pressure),
            "sex": 1 if sex == "Male" else 0, "smoking": int(smoking),
        }])[ALL_FEATURES]

        pipeline = artifacts["best_pipeline"]
        pred = pipeline.predict(patient)[0]
        proba = pipeline.predict_proba(patient)[0, 1]

        risk_level = "🟢 Low" if proba < 0.3 else ("🟡 Moderate" if proba < 0.6 else "🔴 High")

        col1, col2 = st.columns([1, 1])
        with col1:
            st.metric("Predicted Outcome", "Died" if pred == 1 else "Survived")
            st.metric("Predicted Risk of Death", f"{proba:.1%}")
            st.metric("Risk Category", risk_level)

        with col2:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba * 100,
                title={"text": "Risk of Death (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#E63946"},
                    "steps": [
                        {"range": [0, 30], "color": "#c9f7c9"},
                        {"range": [30, 60], "color": "#fff3b0"},
                        {"range": [60, 100], "color": "#ffb3b3"},
                    ],
                }
            ))
            fig_gauge.update_layout(height=300, margin=dict(t=40, b=10))
            st.plotly_chart(fig_gauge, use_container_width=True)

        st.caption(
            "⚠️ This is an academic demonstration model trained on a small, single-cohort research "
            "dataset. It is NOT a validated clinical tool and must never be used for real medical "
            "decisions."
        )
