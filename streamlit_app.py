# ========================================
# HealthStack Analytics
# AI-Powered Population Health Command Center
# ========================================
# This version runs on Streamlit Community Cloud
# connecting to Snowflake via snowflake-connector-python.
# ========================================

import streamlit as st
import snowflake.connector
import pandas as pd
import altair as alt
import json

# -- Page Config --
st.set_page_config(
    page_title="HealthStack - Population Health Command Center",
    page_icon="🏥",
    layout="wide"
)

# -- Custom CSS (adapts to both light and dark themes) --
st.markdown("""
<style>
    .block-container { padding-top: 3.5rem; }

    /* KPI metric cards — theme-adaptive */
    div[data-testid="stMetric"] {
        background-color: var(--secondary-background-color, rgba(28, 58, 92, 0.15));
        border-radius: 8px; padding: 12px 16px;
        border-left: 4px solid #3b82f6;
    }
    div[data-testid="stMetric"] label { font-size: 0.85rem !important; }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-size: 1.8rem !important; font-weight: 700 !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricDelta"] { color: #10b981 !important; }
</style>
""", unsafe_allow_html=True)

# -- Snowflake Connection (with timeout + retry) --
@st.cache_resource
def get_snowflake_connection():
    """Create a cached Snowflake connection with timeout settings."""
    try:
        return snowflake.connector.connect(
            account=st.secrets["snowflake"]["account"],
            user=st.secrets["snowflake"]["user"],
            password=st.secrets["snowflake"]["password"],
            warehouse=st.secrets["snowflake"]["warehouse"],
            database=st.secrets["snowflake"]["database"],
            schema=st.secrets["snowflake"]["schema"],
            role=st.secrets["snowflake"]["role"],
            login_timeout=60,
            network_timeout=60,
        )
    except Exception as e:
        st.error(f"Could not connect to Snowflake: {e}")
        return None

# -- Helper: Run SQL and return DataFrame --
def run_query(sql):
    conn = get_snowflake_connection()
    if conn is None:
        return pd.DataFrame()
    try:
        cur = conn.cursor()
        cur.execute(sql)
        columns = [desc[0] for desc in cur.description]
        data = cur.fetchall()
        return pd.DataFrame(data, columns=columns)
    except (snowflake.connector.errors.DatabaseError,
            snowflake.connector.errors.OperationalError):
        # Connection went stale — clear cache and retry once
        get_snowflake_connection.clear()
        try:
            conn = get_snowflake_connection()
            if conn is None:
                return pd.DataFrame()
            cur = conn.cursor()
            cur.execute(sql)
            columns = [desc[0] for desc in cur.description]
            data = cur.fetchall()
            return pd.DataFrame(data, columns=columns)
        except Exception as e:
            st.error(f"Query error: {str(e)}")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Query error: {str(e)}")
        return pd.DataFrame()

# -- Helper: AI Chat using Cortex COMPLETE --
def ask_ai(question):
    """Use Cortex COMPLETE to answer natural language questions about the data."""
    result = {"text": "", "sql": "", "data": None}

    try:
        # Step 1: Generate SQL from question
        sql_gen = run_query(f"""
            SELECT SNOWFLAKE.CORTEX.COMPLETE(
                'llama3.1-70b',
                'You are a SQL expert for healthcare analytics. Generate a Snowflake SQL query
                 to answer the following question. Only return the SQL, no explanation.

                 Available table: HEALTHCARE_ANALYTICS.ANALYTICS.MEMBER_RISK_PROFILE
                 Columns: member_id, first_name, last_name, age, age_group, gender, plan_type,
                          zip_code, has_diabetes, has_hypertension, has_chf, has_copd, has_ckd,
                          chronic_condition_count, total_claims, er_visits, inpatient_admits,
                          office_visits, total_paid_amount, er_cost, inpatient_cost,
                          latest_hba1c, latest_bnp, latest_egfr, risk_score, risk_tier,
                          composite_risk_score, risk_category, care_gap_count, worsening_trend_count

                 risk_category values: Critical - Immediate Intervention, High Risk - Active Management,
                          Rising Risk - Preventive Outreach, Moderate - Routine Monitoring,
                          Low Risk - Wellness Programs

                 Also available: HEALTHCARE_ANALYTICS.ANALYTICS.CARE_GAPS (member_id, gap_type, gap_severity, measure_category, days_since_action)
                 Also available: HEALTHCARE_ANALYTICS.ANALYTICS.INTERVENTION_RECOMMENDATIONS (member_id, member_name, primary_intervention, outreach_timeline, risk_category, estimated_avoidable_cost)

                 Question: {question}

                 Return ONLY the SQL query, nothing else.'
            ) AS generated_sql
        """)

        if len(sql_gen) > 0:
            generated_sql = sql_gen['GENERATED_SQL'][0].strip()
            generated_sql = generated_sql.replace('```sql', '').replace('```', '').strip()
            result["sql"] = generated_sql

            # Step 2: Execute the SQL
            data = run_query(generated_sql)
            result["data"] = data

            # Step 3: Generate natural language answer
            result_str = data.head(10).to_string()
            nl = run_query(f"""
                SELECT SNOWFLAKE.CORTEX.COMPLETE(
                    'llama3.1-70b',
                    'Based on this data, provide a brief clear answer to: {question}
                     Data: {result_str}
                     Give a concise answer in 2-3 sentences with key numbers.'
                ) AS response
            """)
            if len(nl) > 0:
                result["text"] = nl['RESPONSE'][0]

    except Exception as e:
        result["text"] = f"I had trouble with that question. Try rephrasing. Error: {str(e)}"

    return result


# ========================================
# SIDEBAR
# ========================================
st.sidebar.title("🏥 HealthStack")
st.sidebar.markdown("Population Health Command Center")
st.sidebar.markdown("---")

# Risk Category Filter
risk_cats = run_query("""
    SELECT DISTINCT risk_category
    FROM MEMBER_RISK_PROFILE
    WHERE risk_category IS NOT NULL
    ORDER BY risk_category
""")
if len(risk_cats) > 0:
    selected_risk = st.sidebar.multiselect(
        "Risk Category",
        options=risk_cats['RISK_CATEGORY'].tolist(),
        default=risk_cats['RISK_CATEGORY'].tolist()
    )
else:
    selected_risk = []

# Plan Type Filter
plan_types = run_query("SELECT DISTINCT plan_type FROM MEMBER_RISK_PROFILE ORDER BY plan_type")
if len(plan_types) > 0:
    selected_plans = st.sidebar.multiselect(
        "Plan Type",
        options=plan_types['PLAN_TYPE'].tolist(),
        default=plan_types['PLAN_TYPE'].tolist()
    )
else:
    selected_plans = []

# Condition Filter
st.sidebar.markdown("---")
condition_filter = st.sidebar.multiselect(
    "Chronic Conditions",
    options=["Diabetes", "Hypertension", "Heart Failure", "COPD", "CKD"],
    default=[]
)

# Build WHERE clause
risk_filter = "','".join(selected_risk) if selected_risk else ''
plan_filter = "','".join(selected_plans) if selected_plans else ''
where_clause = f"WHERE risk_category IN ('{risk_filter}') AND plan_type IN ('{plan_filter}')"

condition_sql = ""
for cond in condition_filter:
    col_map = {"Diabetes": "has_diabetes", "Hypertension": "has_hypertension",
               "Heart Failure": "has_chf", "COPD": "has_copd", "CKD": "has_ckd"}
    condition_sql += f" AND {col_map[cond]} = 1"
where_clause += condition_sql

# Sidebar footer
st.sidebar.markdown("---")
st.sidebar.markdown("**HealthStack Analytics** v2.0")
st.sidebar.markdown("Built on Snowflake + Cortex AI")

# ========================================
# NAVIGATION
# ========================================
st.markdown("")  # spacer
nav_cols = st.columns(4)
with nav_cols[0]:
    dash_btn = st.button("📊 Dashboard", use_container_width=True, type="primary" if "nav" not in st.session_state or st.session_state.nav == "Dashboard" else "secondary")
    if dash_btn:
        st.session_state.nav = "Dashboard"
with nav_cols[1]:
    member_btn = st.button("👥 Members", use_container_width=True, type="primary" if st.session_state.get("nav") == "Members" else "secondary")
    if member_btn:
        st.session_state.nav = "Members"
with nav_cols[2]:
    gaps_btn = st.button("⚠️ Care Gaps", use_container_width=True, type="primary" if st.session_state.get("nav") == "Care Gaps" else "secondary")
    if gaps_btn:
        st.session_state.nav = "Care Gaps"
with nav_cols[3]:
    ai_btn = st.button("🤖 AI Assistant", use_container_width=True, type="primary" if st.session_state.get("nav") == "AI" else "secondary")
    if ai_btn:
        st.session_state.nav = "AI"

if "nav" not in st.session_state:
    st.session_state.nav = "Dashboard"

active_tab = st.session_state.nav
st.markdown("---")

# ========================================
# TAB 1: DASHBOARD
# ========================================
if active_tab == "Dashboard":
    st.title("Population Health Dashboard")

    kpis = run_query(f"""
        SELECT
            COUNT(*) AS total_members,
            SUM(CASE WHEN risk_category LIKE 'Critical%' OR risk_category LIKE 'High%' THEN 1 ELSE 0 END) AS high_risk_members,
            ROUND(SUM(total_paid_amount), 0) AS total_cost,
            SUM(er_visits) AS total_er_visits,
            SUM(inpatient_admits) AS total_admits,
            ROUND(AVG(CASE WHEN has_diabetes = 1 THEN latest_hba1c END), 1) AS avg_hba1c,
            ROUND(AVG(chronic_condition_count), 1) AS avg_conditions,
            ROUND(SUM(er_cost + inpatient_cost) / NULLIF(SUM(total_paid_amount), 0) * 100, 1) AS acute_cost_pct
        FROM MEMBER_RISK_PROFILE
        {where_clause}
    """)

    if len(kpis) > 0:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Members", f"{int(kpis['TOTAL_MEMBERS'][0]):,}")
        col2.metric("High/Critical Risk", f"{int(kpis['HIGH_RISK_MEMBERS'][0]):,}",
                    delta=f"{kpis['HIGH_RISK_MEMBERS'][0] / max(kpis['TOTAL_MEMBERS'][0], 1) * 100:.1f}% of pop")
        col3.metric("Total Cost", f"${kpis['TOTAL_COST'][0]:,.0f}")
        col4.metric("ER Visits", f"{int(kpis['TOTAL_ER_VISITS'][0]):,}")

        col5, col6, col7, col8 = st.columns(4)
        col5.metric("IP Admits", f"{int(kpis['TOTAL_ADMITS'][0]):,}")
        col6.metric("Avg HbA1c (Diabetics)", f"{kpis['AVG_HBA1C'][0]}")
        col7.metric("Avg Conditions", f"{kpis['AVG_CONDITIONS'][0]}")
        col8.metric("Acute Care % of Cost", f"{kpis['ACUTE_COST_PCT'][0]}%")

    st.markdown("---")

    # Charts
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("Risk Category Distribution")
        risk_dist = run_query(f"""
            SELECT risk_category, COUNT(*) AS member_count,
                   ROUND(SUM(total_paid_amount), 0) AS total_cost
            FROM MEMBER_RISK_PROFILE {where_clause}
            GROUP BY risk_category ORDER BY member_count DESC
        """)
        if len(risk_dist) > 0:
            donut = alt.Chart(risk_dist).mark_arc(innerRadius=50, outerRadius=120).encode(
                theta=alt.Theta(field="MEMBER_COUNT", type="quantitative"),
                color=alt.Color(field="RISK_CATEGORY", type="nominal",
                               scale=alt.Scale(
                                   domain=['Low Risk - Wellness Programs', 'Moderate - Routine Monitoring',
                                           'Rising Risk - Preventive Outreach', 'High Risk - Active Management',
                                           'Critical - Immediate Intervention'],
                                   range=['#2ecc71', '#f39c12', '#e67e22', '#e74c3c', '#c0392b']
                               ),
                               legend=alt.Legend(title="Risk Category")),
                tooltip=['RISK_CATEGORY', 'MEMBER_COUNT', 'TOTAL_COST']
            ).properties(width=350, height=300)
            st.altair_chart(donut, use_container_width=True)

    with chart_col2:
        st.subheader("Cost by Risk Category")
        if len(risk_dist) > 0:
            cost_bar = alt.Chart(risk_dist).mark_bar(cornerRadiusEnd=4).encode(
                y=alt.Y('RISK_CATEGORY:N', sort='-x', title=''),
                x=alt.X('TOTAL_COST:Q', title='Total Cost ($)', axis=alt.Axis(format='$,.0f')),
                color=alt.Color('RISK_CATEGORY:N',
                               scale=alt.Scale(
                                   domain=['Low Risk - Wellness Programs', 'Moderate - Routine Monitoring',
                                           'Rising Risk - Preventive Outreach', 'High Risk - Active Management',
                                           'Critical - Immediate Intervention'],
                                   range=['#2ecc71', '#f39c12', '#e67e22', '#e74c3c', '#c0392b']
                               ),
                               legend=None),
                tooltip=['RISK_CATEGORY', alt.Tooltip('TOTAL_COST:Q', format='$,.0f')]
            ).properties(width=350, height=300)
            st.altair_chart(cost_bar, use_container_width=True)

    # Condition Prevalence + Cost Breakdown
    chart_col3, chart_col4 = st.columns(2)

    with chart_col3:
        st.subheader("Chronic Condition Prevalence")
        cond_data = run_query(f"""
            SELECT 'Diabetes' AS condition, SUM(has_diabetes) AS members FROM MEMBER_RISK_PROFILE {where_clause}
            UNION ALL SELECT 'Hypertension', SUM(has_hypertension) FROM MEMBER_RISK_PROFILE {where_clause}
            UNION ALL SELECT 'Heart Failure', SUM(has_chf) FROM MEMBER_RISK_PROFILE {where_clause}
            UNION ALL SELECT 'COPD', SUM(has_copd) FROM MEMBER_RISK_PROFILE {where_clause}
            UNION ALL SELECT 'CKD', SUM(has_ckd) FROM MEMBER_RISK_PROFILE {where_clause}
        """)
        if len(cond_data) > 0:
            cond_chart = alt.Chart(cond_data).mark_bar(cornerRadiusEnd=4).encode(
                x=alt.X('MEMBERS:Q', title='Members'),
                y=alt.Y('CONDITION:N', sort='-x', title=''),
                color=alt.Color('CONDITION:N', scale=alt.Scale(scheme='tableau10'), legend=None),
                tooltip=['CONDITION', 'MEMBERS']
            ).properties(height=250)
            st.altair_chart(cond_chart, use_container_width=True)

    with chart_col4:
        st.subheader("Avg Cost Per Member by Risk Tier")
        avg_cost = run_query(f"""
            SELECT risk_category,
                   ROUND(AVG(er_cost), 0) AS avg_er_cost,
                   ROUND(AVG(inpatient_cost), 0) AS avg_ip_cost
            FROM MEMBER_RISK_PROFILE {where_clause}
            GROUP BY risk_category
        """)
        if len(avg_cost) > 0:
            cost_melt = pd.melt(avg_cost, id_vars=['RISK_CATEGORY'],
                                value_vars=['AVG_ER_COST', 'AVG_IP_COST'],
                                var_name='COST_TYPE', value_name='AMOUNT')
            cost_melt['COST_TYPE'] = cost_melt['COST_TYPE'].map({
                'AVG_ER_COST': 'ER Cost', 'AVG_IP_COST': 'Inpatient Cost'
            })
            stacked = alt.Chart(cost_melt).mark_bar(cornerRadiusEnd=2).encode(
                y=alt.Y('RISK_CATEGORY:N', sort='-x', title=''),
                x=alt.X('AMOUNT:Q', title='Avg Cost ($)', stack='zero', axis=alt.Axis(format='$,.0f')),
                color=alt.Color('COST_TYPE:N', scale=alt.Scale(range=['#e74c3c', '#3498db']),
                               legend=alt.Legend(title="Cost Type")),
                tooltip=['RISK_CATEGORY', 'COST_TYPE', alt.Tooltip('AMOUNT:Q', format='$,.0f')]
            ).properties(height=250)
            st.altair_chart(stacked, use_container_width=True)

    # Cost Concentration Table
    st.subheader("Cost Concentration Analysis")
    concentration = run_query(f"""
        SELECT risk_category AS "Risk Category",
            COUNT(*) AS "Members",
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 1) AS "% of Pop",
            ROUND(SUM(total_paid_amount), 0) AS "Total Cost",
            ROUND(SUM(total_paid_amount) * 100.0 / SUM(SUM(total_paid_amount)) OVER(), 1) AS "% of Cost",
            ROUND(AVG(total_paid_amount), 0) AS "Avg Cost/Member",
            ROUND(AVG(er_visits), 1) AS "Avg ER",
            ROUND(AVG(chronic_condition_count), 1) AS "Avg Conditions"
        FROM MEMBER_RISK_PROFILE {where_clause}
        GROUP BY risk_category ORDER BY "Avg Cost/Member" DESC
    """)
    if len(concentration) > 0:
        st.dataframe(concentration, use_container_width=True, hide_index=True)

# ========================================
# TAB 2: MEMBER EXPLORER
# ========================================
if active_tab == "Members":
    st.title("Member Explorer")

    search_col1, search_col2, search_col3 = st.columns([2, 1, 1])
    with search_col1:
        search_term = st.text_input("Search by Member ID or Name", "", key="search")
    with search_col2:
        sort_by = st.selectbox("Sort by", [
            "Risk Score (High to Low)", "Total Cost (High to Low)",
            "ER Visits (High to Low)", "HbA1c (High to Low)"
        ])
    with search_col3:
        show_count = st.selectbox("Show", [25, 50, 100], index=0)

    sort_map = {
        "Risk Score (High to Low)": "COALESCE(composite_risk_score, risk_score) DESC NULLS LAST",
        "Total Cost (High to Low)": "total_paid_amount DESC",
        "ER Visits (High to Low)": "er_visits DESC",
        "HbA1c (High to Low)": "latest_hba1c DESC NULLS LAST"
    }

    search_filter = ""
    if search_term:
        search_filter = f"AND (member_id ILIKE '%{search_term}%' OR first_name ILIKE '%{search_term}%' OR last_name ILIKE '%{search_term}%')"

    members = run_query(f"""
        SELECT member_id AS "ID", first_name || ' ' || last_name AS "Name",
            age AS "Age", gender AS "Sex", plan_type AS "Plan",
            risk_category AS "Risk Category",
            COALESCE(composite_risk_score, risk_score) AS "Risk Score",
            chronic_condition_count AS "Conditions",
            er_visits AS "ER", inpatient_admits AS "IP",
            ROUND(total_paid_amount, 0) AS "Total Cost",
            latest_hba1c AS "HbA1c"
        FROM MEMBER_RISK_PROFILE
        {where_clause} {search_filter}
        ORDER BY {sort_map[sort_by]}
        LIMIT {show_count}
    """)
    if len(members) > 0:
        st.dataframe(members, use_container_width=True, hide_index=True,
                     column_config={
                         "Total Cost": st.column_config.NumberColumn(format="$%d"),
                         "Risk Score": st.column_config.ProgressColumn(min_value=0, max_value=100),
                     })

    # Member Detail
    st.markdown("---")
    st.subheader("Member Detail")
    selected_member = st.text_input("Enter Member ID for details", "", key="detail")

    if selected_member:
        detail = run_query(f"SELECT * FROM MEMBER_RISK_PROFILE WHERE member_id = '{selected_member}'")
        if len(detail) > 0:
            d = detail.iloc[0]
            conditions = []
            if d['HAS_DIABETES'] == 1: conditions.append("Diabetes")
            if d['HAS_HYPERTENSION'] == 1: conditions.append("Hypertension")
            if d['HAS_CHF'] == 1: conditions.append("Heart Failure")
            if d['HAS_COPD'] == 1: conditions.append("COPD")
            if d['HAS_CKD'] == 1: conditions.append("CKD")

            dc1, dc2, dc3 = st.columns(3)
            with dc1:
                st.markdown("**Demographics**")
                st.write(f"**{d['FIRST_NAME']} {d['LAST_NAME']}**")
                st.write(f"Age: {d['AGE']} | Sex: {d['GENDER']}")
                st.write(f"Plan: {d['PLAN_TYPE']}")
            with dc2:
                st.markdown("**Risk Profile**")
                st.write(f"Category: **{d['RISK_CATEGORY']}**")
                st.write(f"Score: {d.get('COMPOSITE_RISK_SCORE', d.get('RISK_SCORE', 'N/A'))}")
                st.write(f"Conditions: {', '.join(conditions) if conditions else 'None'}")
            with dc3:
                st.markdown("**Utilization & Cost**")
                st.write(f"Total Cost: **${d['TOTAL_PAID_AMOUNT']:,.0f}**")
                st.write(f"ER: {d['ER_VISITS']} | IP: {d['INPATIENT_ADMITS']}")
                st.write(f"HbA1c: {d.get('LATEST_HBA1C', 'N/A')} | BNP: {d.get('LATEST_BNP', 'N/A')}")

            # Lab Trends
            try:
                labs = run_query(f"""
                    SELECT test_name, lab_date, result_value, normal_high
                    FROM HEALTHCARE_ANALYTICS.RAW.LAB_RESULTS
                    WHERE member_id = '{selected_member}' ORDER BY test_name, lab_date
                """)
                if len(labs) > 0:
                    st.markdown("---")
                    st.subheader("Lab Value Trends")
                    test_types = labs['TEST_NAME'].unique()
                    trend_cols = st.columns(min(len(test_types), 3))
                    for i, test in enumerate(test_types):
                        with trend_cols[i % 3]:
                            test_data = labs[labs['TEST_NAME'] == test]
                            line = alt.Chart(test_data).mark_line(point=True, color='#e74c3c').encode(
                                x=alt.X('LAB_DATE:T', title='Date'),
                                y=alt.Y('RESULT_VALUE:Q', title=test, scale=alt.Scale(zero=False)),
                                tooltip=['LAB_DATE:T', alt.Tooltip('RESULT_VALUE:Q', format='.1f')]
                            )
                            normal = test_data['NORMAL_HIGH'].iloc[0]
                            rule = alt.Chart(pd.DataFrame({'y': [normal]})).mark_rule(
                                strokeDash=[4,4], color='green').encode(y='y:Q')
                            st.altair_chart((line + rule).properties(title=test, height=200),
                                          use_container_width=True)
            except Exception:
                pass

            # AI Care Summary
            if st.button("🤖 Generate AI Care Summary"):
                with st.spinner("Generating..."):
                    conditions_str = ', '.join(conditions) if conditions else 'None'
                    try:
                        summary = run_query(f"""
                            SELECT SNOWFLAKE.CORTEX.COMPLETE('llama3.1-70b',
                                'You are a clinical care coordinator. Write a brief care summary
                                 and top 3 recommended actions.
                                 Member: {d["FIRST_NAME"]} {d["LAST_NAME"]}, Age: {d["AGE"]}
                                 Conditions: {conditions_str}
                                 ER: {d["ER_VISITS"]}, IP: {d["INPATIENT_ADMITS"]}
                                 Cost: ${d["TOTAL_PAID_AMOUNT"]:,.0f}
                                 HbA1c: {d.get("LATEST_HBA1C","N/A")}, BNP: {d.get("LATEST_BNP","N/A")}
                                 Risk: {d["RISK_CATEGORY"]}'
                            ) AS summary
                        """)
                        if len(summary) > 0:
                            st.markdown(summary['SUMMARY'][0])
                    except Exception as e:
                        st.error(f"Could not generate summary: {e}")
        else:
            st.warning("Member not found.")

# ========================================
# TAB 3: CARE GAPS
# ========================================
if active_tab == "Care Gaps":
    st.title("Care Gap Analysis")
    try:
        gap_kpis = run_query("""
            SELECT COUNT(*) AS total_gaps,
                COUNT(DISTINCT member_id) AS members_with_gaps,
                SUM(CASE WHEN gap_severity = 'Critical' THEN 1 ELSE 0 END) AS critical_gaps,
                SUM(CASE WHEN gap_severity = 'Urgent' THEN 1 ELSE 0 END) AS urgent_gaps
            FROM CARE_GAPS
        """)
        if len(gap_kpis) > 0:
            gk1, gk2, gk3, gk4 = st.columns(4)
            gk1.metric("Total Open Gaps", f"{int(gap_kpis['TOTAL_GAPS'][0]):,}")
            gk2.metric("Members Affected", f"{int(gap_kpis['MEMBERS_WITH_GAPS'][0]):,}")
            gk3.metric("Critical Gaps", f"{int(gap_kpis['CRITICAL_GAPS'][0]):,}")
            gk4.metric("Urgent Gaps", f"{int(gap_kpis['URGENT_GAPS'][0]):,}")

        st.markdown("---")
        gc1, gc2 = st.columns(2)
        with gc1:
            st.subheader("Gaps by Type & Severity")
            gap_detail = run_query("""
                SELECT gap_type, gap_severity, COUNT(*) AS member_count
                FROM CARE_GAPS GROUP BY gap_type, gap_severity ORDER BY gap_type
            """)
            if len(gap_detail) > 0:
                heatmap = alt.Chart(gap_detail).mark_rect(cornerRadius=4).encode(
                    x=alt.X('GAP_SEVERITY:N', title='Severity', sort=['Critical','Urgent','Monitor']),
                    y=alt.Y('GAP_TYPE:N', title=''),
                    color=alt.Color('MEMBER_COUNT:Q', scale=alt.Scale(scheme='reds'), legend=alt.Legend(title="Members")),
                    tooltip=['GAP_TYPE', 'GAP_SEVERITY', 'MEMBER_COUNT']
                ).properties(height=250)
                st.altair_chart(heatmap, use_container_width=True)

        with gc2:
            st.subheader("Gap Distribution")
            by_type = run_query("SELECT gap_type, COUNT(*) AS member_count FROM CARE_GAPS GROUP BY gap_type ORDER BY member_count DESC")
            if len(by_type) > 0:
                gap_donut = alt.Chart(by_type).mark_arc(innerRadius=40, outerRadius=100).encode(
                    theta='MEMBER_COUNT:Q',
                    color=alt.Color('GAP_TYPE:N', scale=alt.Scale(scheme='set2'), legend=alt.Legend(title="Gap Type")),
                    tooltip=['GAP_TYPE', 'MEMBER_COUNT']
                ).properties(height=250)
                st.altair_chart(gap_donut, use_container_width=True)

        st.subheader("Members with Critical Care Gaps")
        critical = run_query("""
            SELECT g.member_id, m.first_name || ' ' || m.last_name AS member_name,
                m.risk_category, g.gap_type, g.gap_severity, g.days_since_action, m.plan_type
            FROM CARE_GAPS g JOIN MEMBER_RISK_PROFILE m ON g.member_id = m.member_id
            WHERE g.gap_severity IN ('Critical','Urgent')
            ORDER BY g.gap_severity, g.days_since_action DESC LIMIT 50
        """)
        if len(critical) > 0:
            st.dataframe(critical, use_container_width=True, hide_index=True)
    except Exception:
        st.info("Care Gaps table not available. Run Phase 3 SQL to create it.")

# ========================================
# TAB 4: AI ASSISTANT
# ========================================
if active_tab == "AI":
    st.title("🤖 AI Health Analytics Assistant")
    st.markdown("Ask questions about your population in plain English.")

    # -- Suggested questions --
    st.markdown("**Try asking:**")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("- How many members are in each risk category?")
        st.markdown("- What is the total cost for high risk members?")
        st.markdown("- Show me diabetic members with more than 2 ER visits")
    with c2:
        st.markdown("- What is the average HbA1c by risk category?")
        st.markdown("- Which plan type has the highest cost per member?")
        st.markdown("- How many members have both diabetes and heart failure?")

    st.markdown("---")

    # -- Input box (text_input + button — more reliable than chat_input) --
    input_col, btn_col = st.columns([5, 1])
    with input_col:
        user_question = st.text_input(
            "Your question",
            placeholder="Ask about your population...",
            label_visibility="collapsed",
            key="ai_question"
        )
    with btn_col:
        ask_clicked = st.button("Ask 🚀", use_container_width=True, type="primary")

    # -- Conversation history --
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Process the question when button is clicked (or Enter pressed)
    if ask_clicked and user_question:
        st.session_state.messages.append({"role": "user", "content": user_question})

        with st.spinner("Analyzing your question..."):
            result = ask_ai(user_question)

        st.session_state.messages.append({
            "role": "assistant",
            "content": result.get("text", ""),
            "sql": result.get("sql", ""),
            "data": result.get("data")
        })

    # -- Display conversation (newest first) --
    for message in reversed(st.session_state.messages):
        if message["role"] == "user":
            st.markdown(f"**🧑 You:** {message['content']}")
        else:
            st.markdown(f"**🤖 Assistant:** {message['content']}")
            if "sql" in message and message["sql"]:
                with st.expander("View SQL"):
                    st.code(message["sql"], language="sql")
            if "data" in message and message["data"] is not None and len(message["data"]) > 0:
                st.dataframe(message["data"].head(20), hide_index=True)

                # Auto-chart small results
                if len(message["data"]) <= 10 and len(message["data"].columns) >= 2:
                    try:
                        cols = message["data"].columns.tolist()
                        str_col = next((c for c in cols if message["data"][c].dtype == 'object'), None)
                        num_col = next((c for c in cols if message["data"][c].dtype in ['int64','float64']), None)
                        if str_col and num_col:
                            chart = alt.Chart(message["data"]).mark_bar(cornerRadiusEnd=4).encode(
                                y=alt.Y(f'{str_col}:N', sort='-x', title=''),
                                x=alt.X(f'{num_col}:Q', title=num_col),
                                color=alt.Color(f'{str_col}:N', legend=None, scale=alt.Scale(scheme='tableau10'))
                            ).properties(height=min(len(message["data"]) * 40, 300))
                            st.altair_chart(chart, use_container_width=True)
                    except Exception:
                        pass
        st.markdown("---")
