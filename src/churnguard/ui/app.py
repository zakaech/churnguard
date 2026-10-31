"""Interface Streamlit de ChurnGuard.

Lancement : `python tasks.py ui` (l'API doit tourner : `python tasks.py api`).
Variables d'environnement : API_URL (défaut http://127.0.0.1:8000), CHURNGUARD_API_KEY.
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from churnguard.labels import COMMON_VALUES, FEATURE_LABELS, VALUE_LABELS, format_number
from churnguard.ui.client import (
    INTERNET_OPTIONS,
    ApiClient,
    ApiError,
    build_customer,
    format_detail,
    results_table,
    risk_distribution,
)

BLUE, ORANGE = "#2a78d6", "#eb6834"
RISK_DISPLAY = {
    "élevé": ("🔴", "Risque élevé", st.error),
    "moyen": ("🟠", "Risque moyen", st.warning),
    "faible": ("🟢", "Risque faible", st.success),
}
YES_NO = ["No", "Yes"]
OPTIONS = {
    "gender": ["Female", "Male"],
    "Partner": YES_NO,
    "Dependents": YES_NO,
    "PhoneService": ["Yes", "No"],
    "MultipleLines": YES_NO,
    "InternetService": ["Fiber optic", "DSL", "No"],
    **dict.fromkeys(INTERNET_OPTIONS, YES_NO),
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
}


def label_of(feature: str):
    return lambda v: VALUE_LABELS.get(feature, {}).get(v, COMMON_VALUES.get(v, v))


def percent(p: float) -> str:
    return f"{format_number(100 * p, 1)} %"


@st.cache_resource
def get_client() -> ApiClient:
    return ApiClient()


def select(feature: str, disabled: bool = False) -> str:
    return st.selectbox(
        FEATURE_LABELS[feature],
        OPTIONS[feature],
        format_func=label_of(feature),
        disabled=disabled,
        key=feature,
    )


def factors_chart(factors: list[dict]) -> alt.Chart:
    data = pd.DataFrame(
        {
            "facteur": [f"{f['label']} : {f['value_label']}" for f in factors],
            "impact": [f["impact_points"] for f in factors],
            "sens": [
                "augmente le risque" if f["impact_points"] > 0 else "diminue le risque"
                for f in factors
            ],
        }
    )
    return (
        alt.Chart(data)
        .mark_bar(cornerRadiusEnd=4, height=18)
        .encode(
            x=alt.X("impact:Q", title="impact sur la probabilité (points)"),
            y=alt.Y("facteur:N", sort=None, title=None),
            color=alt.Color(
                "sens:N",
                scale=alt.Scale(
                    domain=["augmente le risque", "diminue le risque"], range=[ORANGE, BLUE]
                ),
                legend=alt.Legend(title=None, orient="bottom"),
            ),
            tooltip=["facteur", alt.Tooltip("impact:Q", format="+.1f"), "sens"],
        )
    )


def show_result(result: dict) -> None:
    icon, title, box = RISK_DISPLAY[result["risk_level"]]
    decision = "client signalé" if result["churn_predicted"] else "client non signalé"
    st.metric("Probabilité de départ", percent(result["churn_probability"]))
    box(
        f"**{title}** — {decision} (seuil de décision : {percent(result['threshold'])})",
        icon=icon,
    )
    st.subheader("Principaux facteurs")
    st.altair_chart(factors_chart(result["top_factors"]), width="stretch")
    for f in result["top_factors"]:
        st.markdown(f"- {f['text']}")


# --- Pages ---------------------------------------------------------------------------------


def page_client(client: ApiClient) -> None:
    st.header("Évaluer un client")
    st.caption("Renseignez le profil du client, puis lancez l'évaluation.")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Profil**")
        gender = select("gender")
        senior = st.checkbox(FEATURE_LABELS["SeniorCitizen"], key="SeniorCitizen")
        partner = select("Partner")
        dependents = select("Dependents")
    with col2:
        st.markdown("**Contrat et facturation**")
        tenure = st.number_input(
            f"{FEATURE_LABELS['tenure']} (mois)", min_value=0, max_value=1200, value=2, step=1
        )
        contract = select("Contract")
        paperless = select("PaperlessBilling")
        payment = select("PaymentMethod")
        monthly = st.number_input(
            FEATURE_LABELS["MonthlyCharges"], min_value=0.01, value=85.5, step=1.0
        )
        total = st.number_input(
            FEATURE_LABELS["TotalCharges"],
            min_value=0.0,
            value=float(round(tenure * monthly, 2)),
            step=10.0,
            help="Par défaut : ancienneté × facture mensuelle.",
        )
    with col3:
        st.markdown("**Services**")
        phone = select("PhoneService")
        lines = select("MultipleLines", disabled=phone == "No")
        internet = select("InternetService")
        options = {o: select(o, disabled=internet == "No") for o in INTERNET_OPTIONS}

    if st.button("Évaluer le risque", type="primary"):
        customer = build_customer(
            {
                "gender": gender,
                "SeniorCitizen": senior,
                "Partner": partner,
                "Dependents": dependents,
                "tenure": int(tenure),
                "PhoneService": phone,
                "MultipleLines": lines,
                "InternetService": internet,
                **options,
                "Contract": contract,
                "PaperlessBilling": paperless,
                "PaymentMethod": payment,
                "MonthlyCharges": float(monthly),
                "TotalCharges": float(total),
            }
        )
        try:
            with st.spinner("Évaluation en cours…"):
                show_result(client.predict(customer))
        except ApiError as e:
            st.error(format_detail(e.detail), icon="⚠️")


def page_csv(client: ApiClient, health: dict) -> None:
    st.header("Évaluer un fichier de clients")
    st.caption(
        "CSV au format brut du fichier Telco (mêmes colonnes ; `customerID` conseillé). "
        "Les colonnes en trop, comme `Churn`, sont ignorées."
    )
    uploaded = st.file_uploader("Fichier CSV", type=["csv"])
    with_factors = st.checkbox(
        "Calculer les facteurs de chaque client (plus lent)",
        help=f"Lots de {health['max_batch_size_with_factors']} clients avec facteurs, "
        f"{health['max_batch_size']} sans.",
    )
    if uploaded is not None and st.button("Analyser le fichier", type="primary"):
        size = health["max_batch_size_with_factors" if with_factors else "max_batch_size"]
        try:
            with st.spinner("Analyse en cours…"):
                predictions, ignored = client.predict_csv(uploaded.getvalue(), with_factors, size)
        except ApiError as e:
            st.error(format_detail(e.detail), icon="⚠️")
            return
        st.session_state["batch"] = {"name": uploaded.name, "predictions": predictions}
        if ignored:
            st.info(f"Colonnes ignorées : {', '.join(ignored)}")

    batch = st.session_state.get("batch")
    if not batch:
        return
    st.subheader(f"Résultats — {batch['name']} ({len(batch['predictions'])} clients)")
    dist = risk_distribution(batch["predictions"])
    for col, row in zip(st.columns(3), dist.itertuples(), strict=True):
        icon, title, _ = RISK_DISPLAY[row.niveau_risque]
        col.metric(f"{icon} {title}", f"{row.clients}", f"{format_number(100 * row.part, 1)} %",
                   delta_color="off")  # fmt: skip
    table = results_table(batch["predictions"])
    st.dataframe(table, width="stretch", hide_index=True)
    st.download_button(
        "Télécharger les résultats (CSV)",
        table.to_csv(index=False).encode("utf-8-sig"),
        file_name="churnguard_resultats.csv",
        mime="text/csv",
    )


def page_analysis(client: ApiClient) -> None:
    st.header("Analyse")
    try:
        info = client.model_info()
    except ApiError as e:
        st.error(format_detail(e.detail), icon="⚠️")
        return

    st.subheader("Performances du modèle")
    st.caption(
        f"Modèle {info['model_name']} v{info['model_version']} — seuil "
        f"{percent(info['threshold'])} ({info['threshold_rule']})."
    )
    rows = {
        "validation": info["validation_metrics"],
        "test (évaluation unique)": info["test_metrics"],
    }
    metrics = pd.DataFrame(
        {
            name: {
                "précision": m["precision"],
                "rappel": m["recall"],
                "F1": m["f1"],
                "ROC-AUC": m["roc_auc"],
                "PR-AUC": m["pr_auc"],
            }
            for name, m in rows.items()
            if m
        }
    ).T
    st.dataframe(metrics.style.format("{:.3f}"), width="stretch")

    if info["global_importance"]:
        st.subheader("Importance globale des variables (SHAP, validation)")
        imp = pd.DataFrame(info["global_importance"])
        chart = (
            alt.Chart(imp)
            .mark_bar(color=BLUE, cornerRadiusEnd=4, height=14)
            .encode(
                x=alt.X("mean_abs_points:Q", title="impact moyen (points de probabilité)"),
                y=alt.Y("label:N", sort="-x", title=None),
                tooltip=["label", alt.Tooltip("mean_abs_points:Q", format=".1f")],
            )
        )
        st.altair_chart(chart, width="stretch")

    st.subheader("Répartition des risques du dernier fichier importé")
    batch = st.session_state.get("batch")
    if not batch:
        st.info("Importez un fichier dans la page « Import CSV » pour voir sa répartition.")
        return
    dist = risk_distribution(batch["predictions"])
    chart = (
        alt.Chart(dist)
        .mark_bar(color=BLUE, cornerRadiusEnd=4, size=40)
        .encode(
            x=alt.X("niveau_risque:N", sort=["élevé", "moyen", "faible"], title="niveau de risque"),
            y=alt.Y("clients:Q", title="clients"),
            tooltip=["niveau_risque", "clients", alt.Tooltip("part:Q", format=".1%")],
        )
    )
    st.altair_chart(chart, width="stretch")
    st.caption(f"Fichier : {batch['name']} — {len(batch['predictions'])} clients.")


def main() -> None:
    st.set_page_config(page_title="ChurnGuard", page_icon="📉", layout="wide")
    st.sidebar.title("ChurnGuard")
    client = get_client()
    try:
        health = client.health()
    except ApiError as e:
        st.error(f"API injoignable ou en erreur : {e}", icon="⚠️")
        st.stop()
    st.sidebar.caption(f"API connectée — modèle {health['model_name']} v{health['model_version']}")
    page = st.sidebar.radio("Page", ["Client", "Import CSV", "Analyse"])
    if page == "Client":
        page_client(client)
    elif page == "Import CSV":
        page_csv(client, health)
    else:
        page_analysis(client)


main()
