import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime
from fpdf import FPDF
from openai import OpenAI

DB_PATH = "finance_app.db"
MODEL_NAME = "gpt-4.1-mini"

st.set_page_config(page_title="Retail Finance Chatbot", layout="wide")

if "show_main_app" not in st.session_state:
    st.session_state.show_main_app = False


# -----------------------------
# Database setup
# -----------------------------
def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def init_db():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS budget (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        fiscal_year TEXT,
        month TEXT,
        category TEXT,
        amount REAL
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS transactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_name TEXT,
        transaction_type TEXT,
        date TEXT,
        description TEXT,
        category TEXT,
        amount REAL
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS reports (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        report_month TEXT,
        report_year TEXT,
        report_text TEXT,
        created_at TEXT
    )
    """)

    conn.commit()
    conn.close()


# -----------------------------
# File processing
# -----------------------------
def read_uploaded_file(uploaded_file):
    file_name = uploaded_file.name.lower()
    if file_name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif file_name.endswith(".xlsx"):
        return pd.read_excel(uploaded_file)
    else:
        raise ValueError("Only CSV and XLSX files are supported.")


def normalize_budget_df(df):
    df.columns = [c.strip().lower() for c in df.columns]

    required = {"fiscal_year", "month", "category", "amount"}
    if not required.issubset(set(df.columns)):
        raise ValueError(
            "Budget file must include columns: fiscal_year, month, category, amount"
        )

    df = df[["fiscal_year", "month", "category", "amount"]].copy()
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df.dropna(subset=["amount"])
    df["fiscal_year"] = df["fiscal_year"].astype(str)
    df["month"] = df["month"].astype(str).str.strip()
    df["category"] = df["category"].astype(str).str.strip()
    return df


def normalize_transaction_df(df, transaction_type):
    df.columns = [c.strip().lower() for c in df.columns]

    required = {"date", "description", "category", "amount"}
    if not required.issubset(set(df.columns)):
        raise ValueError(
            f"{transaction_type.title()} file must include columns: date, description, category, amount"
        )

    df = df[["date", "description", "category", "amount"]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df.dropna(subset=["date", "amount"])
    df["description"] = df["description"].astype(str).str.strip()
    df["category"] = df["category"].astype(str).str.strip()
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    df["transaction_type"] = transaction_type
    return df


def save_budget_to_db(df):
    conn = get_conn()

    conn.execute("DELETE FROM budget")
    conn.commit()

    df.to_sql("budget", conn, if_exists="append", index=False)
    conn.close()


def save_transactions_to_db(df, file_name):
    conn = get_conn()

    conn.execute("DELETE FROM transactions WHERE file_name = ?", (file_name,))
    conn.commit()

    df = df.copy()
    df["file_name"] = file_name
    df = df[["file_name", "transaction_type", "date", "description", "category", "amount"]]

    df.to_sql("transactions", conn, if_exists="append", index=False)
    conn.close()


def load_sample_data():
    budget_df = pd.read_csv("data/budget_2025_sample.csv")
    deposits_df = pd.read_csv("data/deposits_2025_sample.csv")
    invoices_df = pd.read_csv("data/invoices_2025_sample.csv")

    budget_df = normalize_budget_df(budget_df)
    deposits_df = normalize_transaction_df(deposits_df, "deposit")
    invoices_df = normalize_transaction_df(invoices_df, "invoice")

    save_budget_to_db(budget_df)
    save_transactions_to_db(deposits_df, "deposits_2025_sample.csv")
    save_transactions_to_db(invoices_df, "invoices_2025_sample.csv")


def has_data():
    conn = get_conn()
    budget_count = pd.read_sql_query("SELECT COUNT(*) AS count FROM budget", conn).iloc[0]["count"]
    transaction_count = pd.read_sql_query("SELECT COUNT(*) AS count FROM transactions", conn).iloc[0]["count"]
    conn.close()
    return budget_count > 0 and transaction_count > 0


# -----------------------------
# Finance calculations
# -----------------------------
def get_monthly_revenue(month, year):
    conn = get_conn()
    query = """
    SELECT COALESCE(SUM(amount), 0) AS revenue
    FROM transactions
    WHERE transaction_type = 'deposit'
      AND strftime('%m', date) = ?
      AND strftime('%Y', date) = ?
    """
    revenue = pd.read_sql_query(
        query, conn, params=(f"{int(month):02d}", str(year))
    ).iloc[0]["revenue"]
    conn.close()
    return float(revenue or 0)


def get_monthly_expenses(month, year):
    conn = get_conn()
    query = """
    SELECT COALESCE(SUM(amount), 0) AS expenses
    FROM transactions
    WHERE transaction_type = 'invoice'
      AND strftime('%m', date) = ?
      AND strftime('%Y', date) = ?
    """
    expenses = pd.read_sql_query(
        query, conn, params=(f"{int(month):02d}", str(year))
    ).iloc[0]["expenses"]
    conn.close()
    return float(expenses or 0)


def get_monthly_budget(month_name, fiscal_year):
    conn = get_conn()
    query = """
    SELECT COALESCE(SUM(amount), 0) AS budget_total
    FROM budget
    WHERE lower(month) = lower(?)
      AND fiscal_year = ?
    """
    budget_total = pd.read_sql_query(
        query, conn, params=(month_name, str(fiscal_year))
    ).iloc[0]["budget_total"]
    conn.close()
    return float(budget_total or 0)


def get_top_expense_categories(month, year):
    conn = get_conn()
    query = """
    SELECT category, SUM(amount) AS total
    FROM transactions
    WHERE transaction_type = 'invoice'
      AND strftime('%m', date) = ?
      AND strftime('%Y', date) = ?
    GROUP BY category
    ORDER BY total DESC
    LIMIT 5
    """
    df = pd.read_sql_query(query, conn, params=(f"{int(month):02d}", str(year)))
    conn.close()
    return df


def get_recent_reports(limit=3):
    conn = get_conn()
    query = """
    SELECT report_month, report_year, report_text, created_at
    FROM reports
    ORDER BY id DESC
    LIMIT ?
    """
    df = pd.read_sql_query(query, conn, params=(limit,))
    conn.close()
    return df


def save_report(month_name, year, report_text):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO reports (report_month, report_year, report_text, created_at)
        VALUES (?, ?, ?, ?)
    """, (month_name, str(year), report_text, datetime.now().isoformat()))
    conn.commit()
    conn.close()


def build_financial_summary(month, year):
    month_name = datetime(year, month, 1).strftime("%B")
    revenue = get_monthly_revenue(month, year)
    expenses = get_monthly_expenses(month, year)
    profit = revenue - expenses
    budget_total = get_monthly_budget(month_name, year)
    variance = budget_total - expenses
    top_expenses = get_top_expense_categories(month, year)

    top_expense_text = "No invoice data available."
    if not top_expenses.empty:
        lines = []
        for _, row in top_expenses.iterrows():
            lines.append(f"- {row['category']}: ${row['total']:,.2f}")
        top_expense_text = "\n".join(lines)

    summary = f"""
Month: {month_name} {year}

Revenue (from deposits): ${revenue:,.2f}
Expenses (from invoices): ${expenses:,.2f}
Profit: ${profit:,.2f}

Budgeted expenses for {month_name}: ${budget_total:,.2f}
Budget variance (budget - actual expenses): ${variance:,.2f}

Top expense categories:
{top_expense_text}
""".strip()

    return {
        "month_name": month_name,
        "year": year,
        "revenue": revenue,
        "expenses": expenses,
        "profit": profit,
        "budget_total": budget_total,
        "variance": variance,
        "top_expenses_df": top_expenses,
        "summary_text": summary
    }


# -----------------------------
# OpenAI
# -----------------------------
def get_openai_client():
    if "OPENAI_API_KEY" not in st.secrets:
        raise ValueError(
            "OPENAI_API_KEY is missing. Add it to Streamlit secrets before using the chatbot."
        )
    return OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


def ask_openai(user_question, summary_text, recent_reports_text):
    client = get_openai_client()

    system_prompt = f"""
You are a financial analyst assistant for a small retail business.

Rules:
- Use only the data provided below.
- Do not invent numbers.
- Be clear, concise, and practical.
- If the data is missing, say so.
- If the user asks for a report, provide a professional monthly report.
- Keep responses readable for a non-technical small business owner.

Current financial summary:
{summary_text}

Recent saved reports:
{recent_reports_text}
""".strip()

    response = client.responses.create(
        model=MODEL_NAME,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_question}
        ]
    )

    return response.output_text


# -----------------------------
# PDF helper
# -----------------------------
def create_pdf_report(title, report_text):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    pdf.multi_cell(0, 10, title)
    pdf.ln(5)

    pdf.set_font("Helvetica", size=11)

    clean_text = (
        report_text.replace("’", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("–", "-")
        .replace("—", "-")
    )

    for line in clean_text.split("\n"):
        pdf.multi_cell(0, 8, line)

    return bytes(pdf.output())


# -----------------------------
# Welcome page
# -----------------------------
def show_welcome_page():
    st.title("Welcome to the Retail Finance Chatbot")
    st.subheader("Analyze retail financial data and generate AI-powered insights")

    st.markdown("""
    ### What this app does
    - Upload budget, deposit, and invoice files
    - Generate monthly financial summaries
    - Ask AI questions about your financial performance
    - Download AI-generated reports as PDF

    ### How to use it
    1. Click **Launch App**
    2. Go to **Upload Data**
    3. Upload your budget, deposit, and invoice files or load the sample data
    4. Open **Monthly Report** to generate a summary
    5. Use **Chatbot** to ask financial questions

    ### Accepted file types
    - CSV
    - XLSX
    """)

    if st.button("Launch App"):
        st.session_state.show_main_app = True
        st.rerun()


# -----------------------------
# App UI
# -----------------------------
init_db()

if not st.session_state.show_main_app:
    show_welcome_page()
    st.stop()

top_col1, top_col2 = st.columns([6, 1])
with top_col1:
    st.title("Retail Finance Chatbot")
    st.caption("Upload budget, invoice, and deposit files, then ask financial questions.")
with top_col2:
    if st.button("Back to Welcome"):
        st.session_state.show_main_app = False
        st.rerun()

tab1, tab2, tab3 = st.tabs(["Upload Data", "Monthly Report", "Chatbot"])

with tab1:
    st.subheader("Quick Start")
    if st.button("Load Sample Data"):
        try:
            load_sample_data()
            st.success("Sample data loaded successfully.")
        except Exception as e:
            st.error(f"Could not load sample data: {e}")

    st.subheader("1. Upload Yearly Budget")
    budget_file = st.file_uploader("Upload budget CSV/XLSX", type=["csv", "xlsx"], key="budget")
    if budget_file is not None:
        try:
            budget_df = read_uploaded_file(budget_file)
            budget_df = normalize_budget_df(budget_df)
            save_budget_to_db(budget_df)
            st.success("Budget uploaded successfully.")
            st.dataframe(budget_df.head())
        except Exception as e:
            st.error(f"Budget upload failed: {e}")

    st.subheader("2. Upload Deposit File")
    deposit_file = st.file_uploader("Upload deposits CSV/XLSX", type=["csv", "xlsx"], key="deposit")
    if deposit_file is not None:
        try:
            deposit_df = read_uploaded_file(deposit_file)
            deposit_df = normalize_transaction_df(deposit_df, "deposit")
            save_transactions_to_db(deposit_df, deposit_file.name)
            st.success("Deposit file uploaded successfully.")
            st.dataframe(deposit_df.head())
        except Exception as e:
            st.error(f"Deposit upload failed: {e}")

    st.subheader("3. Upload Invoice File")
    invoice_file = st.file_uploader("Upload invoices CSV/XLSX", type=["csv", "xlsx"], key="invoice")
    if invoice_file is not None:
        try:
            invoice_df = read_uploaded_file(invoice_file)
            invoice_df = normalize_transaction_df(invoice_df, "invoice")
            save_transactions_to_db(invoice_df, invoice_file.name)
            st.success("Invoice file uploaded successfully.")
            st.dataframe(invoice_df.head())
        except Exception as e:
            st.error(f"Invoice upload failed: {e}")

with tab2:
    st.subheader("Generate Monthly Report")

    if not has_data():
        st.warning("Please upload your files or load the sample data first.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            month = st.number_input("Month (1-12)", min_value=1, max_value=12, value=datetime.now().month)
        with col2:
            year = st.number_input("Year", min_value=2020, max_value=2100, value=datetime.now().year)

        if st.button("Generate Report"):
            try:
                result = build_financial_summary(month, year)
                st.markdown(f"### {result['month_name']} {result['year']} Financial Summary")
                st.write(f"**Revenue:** ${result['revenue']:,.2f}")
                st.write(f"**Expenses:** ${result['expenses']:,.2f}")
                st.write(f"**Profit:** ${result['profit']:,.2f}")
                st.write(f"**Budgeted Expenses:** ${result['budget_total']:,.2f}")
                st.write(f"**Budget Variance (Budget - Actual Expenses):** ${result['variance']:,.2f}")

                st.markdown("#### Top Expense Categories")
                if result["top_expenses_df"].empty:
                    st.info("No invoice data available for this month.")
                else:
                    st.dataframe(result["top_expenses_df"])

                recent_reports = get_recent_reports(limit=3)
                recent_reports_text = ""
                if not recent_reports.empty:
                    for _, row in recent_reports.iterrows():
                        recent_reports_text += f"\n[{row['report_month']} {row['report_year']}]\n{row['report_text']}\n"

                prompt = f"Create a professional monthly financial report for {result['month_name']} {result['year']}."

                with st.spinner("Generating AI report..."):
                    report_text = ask_openai(prompt, result["summary_text"], recent_reports_text)

                st.markdown("#### AI-Generated Report")
                st.write(report_text)

                pdf_title = f"{result['month_name']} {result['year']} Financial Report"
                pdf_bytes = create_pdf_report(pdf_title, report_text)

                st.download_button(
                    label="Download Report as PDF",
                    data=pdf_bytes,
                    file_name=f"{result['month_name']}_{result['year']}_report.pdf",
                    mime="application/pdf"
                )

                save_report(result["month_name"], result["year"], report_text)
                st.success("Report generated and saved to memory.")
            except Exception as e:
                st.error(f"Could not generate report: {e}")

with tab3:
    st.subheader("Ask the Chatbot")

    if not has_data():
        st.warning("Please upload your files or load the sample data first.")
    else:
        month = st.number_input("Chat month (1-12)", min_value=1, max_value=12, value=datetime.now().month, key="chat_month")
        year = st.number_input("Chat year", min_value=2020, max_value=2100, value=datetime.now().year, key="chat_year")
        user_question = st.text_area("Ask a question", placeholder="Example: Are we over budget this month?")

        if st.button("Ask"):
            try:
                result = build_financial_summary(month, year)
                recent_reports = get_recent_reports(limit=3)
                recent_reports_text = ""
                if not recent_reports.empty:
                    for _, row in recent_reports.iterrows():
                        recent_reports_text += f"\n[{row['report_month']} {row['report_year']}]\n{row['report_text']}\n"

                with st.spinner("Analyzing your question..."):
                    answer = ask_openai(user_question, result["summary_text"], recent_reports_text)

                st.markdown("#### Answer")
                st.write(answer)
            except Exception as e:
                st.error(f"Chat failed: {e}")
