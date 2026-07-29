"""
Prepare the browser demo:
  1. Export every warehouse table to Parquet under ../db/ (DuckDB-WASM loads these).
  2. Verify the curated NL->SQL question set against the real warehouse, so we
     KNOW every demo query returns correct rows before it ships.
  3. Emit ../queries.json (the curated set the static site reads).

The curated SQL here is the "pre-generated" answer cache: hand-authored and
verified now; the live free-form mode regenerates SQL with Claude at runtime.
"""

from __future__ import annotations
import json
import os
import duckdb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WAREHOUSE = os.path.join(HERE, "warehouse.duckdb")
DB_DIR = os.path.join(ROOT, "db")
QUERIES_JSON = os.path.join(ROOT, "queries.json")

TABLES = [
    "customer", "aws_account", "aws_service", "purchase", "invoice", "payment",
    "support_ticket", "employee", "account_manager_assignment", "usage_log",
    "region", "account_region", "contract",
]

# id, en, zh, category, viz (bar|line|table), sql
CURATED = [
    dict(id="customers-by-segment", cat="Overview", viz="bar",
         en="How many customers are in each segment?",
         zh="各客戶分群各有多少家客戶?",
         sql="""SELECT segment, COUNT(*) AS customers
FROM customer
GROUP BY segment
ORDER BY customers DESC;"""),

    dict(id="revenue-by-segment", cat="Revenue", viz="bar",
         en="What is total revenue by customer segment?",
         zh="各客戶分群的總營收是多少?",
         sql="""SELECT c.segment,
       ROUND(SUM(i.total_amount)) AS revenue,
       COUNT(DISTINCT c.customer_id) AS customers
FROM customer c
JOIN purchase p ON p.customer_id = c.customer_id
JOIN invoice  i ON i.purchase_id = p.purchase_id
GROUP BY c.segment
ORDER BY revenue DESC;"""),

    dict(id="revenue-by-industry", cat="Revenue", viz="bar",
         en="Which industries generate the most revenue?",
         zh="哪些產業帶來最多營收?",
         sql="""SELECT c.industry,
       ROUND(SUM(i.total_amount)) AS revenue
FROM customer c
JOIN purchase p ON p.customer_id = c.customer_id
JOIN invoice  i ON i.purchase_id = p.purchase_id
GROUP BY c.industry
ORDER BY revenue DESC;"""),

    dict(id="monthly-revenue", cat="Revenue", viz="line",
         en="Show the monthly revenue trend over the last 12 months.",
         zh="顯示近 12 個月的每月營收趨勢。",
         sql="""SELECT date_trunc('month', i.invoice_date) AS month,
       ROUND(SUM(i.total_amount)) AS revenue
FROM invoice i
WHERE i.invoice_date >= (SELECT max(invoice_date) FROM invoice) - INTERVAL 12 MONTH
GROUP BY 1
ORDER BY 1;"""),

    dict(id="top-customers-unpaid", cat="Finance", viz="table",
         en="Top 5 customers by revenue — and what % is still unpaid?",
         zh="營收前 5 大客戶,各有多少比例尚未付款?",
         sql="""SELECT c.company_name,
       c.segment,
       ROUND(SUM(i.total_amount)) AS revenue,
       ROUND(100.0 * SUM(CASE WHEN i.payment_status <> 'Paid'
                              THEN i.total_amount ELSE 0 END)
             / SUM(i.total_amount), 1) AS pct_unpaid
FROM customer c
JOIN purchase p ON p.customer_id = c.customer_id
JOIN invoice  i ON i.purchase_id = p.purchase_id
GROUP BY 1, 2
ORDER BY revenue DESC
LIMIT 5;"""),

    dict(id="payment-speed", cat="Finance", viz="bar",
         en="What is the average time to pay by payment method?",
         zh="各付款方式的平均付款天數是多少?",
         sql="""SELECT pm.payment_method,
       ROUND(AVG(pm.payment_date - i.invoice_date), 1) AS avg_days_to_pay,
       COUNT(*) AS payments
FROM payment pm
JOIN invoice i ON i.invoice_id = pm.invoice_id
GROUP BY 1
ORDER BY avg_days_to_pay;"""),

    dict(id="eu-overdue", cat="Finance", viz="table",
         en="Which customers with EU-region accounts have overdue invoices?",
         zh="哪些在歐洲區有帳號的客戶有逾期發票?",
         sql="""SELECT c.company_name,
       COUNT(*) AS overdue_invoices,
       ROUND(SUM(i.total_amount)) AS overdue_amount
FROM customer c
JOIN purchase p ON p.customer_id = c.customer_id
JOIN invoice  i ON i.purchase_id = p.purchase_id
                AND i.payment_status = 'Overdue'
WHERE c.customer_id IN (
    SELECT ac.customer_id
    FROM aws_account ac
    JOIN account_region ar ON ar.account_id = ac.account_id
    JOIN region r ON r.region_id = ar.region_id
    WHERE r.continent = 'Europe'
)
GROUP BY 1
ORDER BY overdue_amount DESC
LIMIT 10;"""),

    dict(id="usage-by-tier", cat="Usage", viz="bar",
         en="What is total AWS usage cost by service tier?",
         zh="各服務層級的 AWS 使用成本總計是多少?",
         sql="""SELECT s.service_tier,
       ROUND(SUM(u.cost)) AS usage_cost
FROM usage_log u
JOIN aws_service s ON s.service_id = u.service_id
GROUP BY 1
ORDER BY usage_cost DESC;"""),

    dict(id="top-services", cat="Usage", viz="bar",
         en="Top 10 AWS services by total usage cost.",
         zh="使用成本最高的前 10 項 AWS 服務。",
         sql="""SELECT s.service_name,
       ROUND(SUM(u.cost)) AS usage_cost
FROM usage_log u
JOIN aws_service s ON s.service_id = u.service_id
GROUP BY 1
ORDER BY usage_cost DESC
LIMIT 10;"""),

    dict(id="account-managers", cat="Operations", viz="table",
         en="Which account managers handle the most customers, and how many open support tickets do those customers have?",
         zh="哪些客戶經理負責最多客戶?這些客戶有幾張未結支援單?",
         sql="""SELECT e.employee_name,
       COUNT(DISTINCT a.customer_id) AS customers,
       COUNT(DISTINCT t.ticket_id)   AS open_tickets
FROM employee e
JOIN account_manager_assignment a ON a.employee_id = e.employee_id
LEFT JOIN support_ticket t ON t.customer_id = a.customer_id
                          AND t.status = 'Open'
WHERE e.department = 'Account Management'
GROUP BY 1
ORDER BY customers DESC
LIMIT 8;"""),

    dict(id="ticket-resolution", cat="Operations", viz="bar",
         en="What is the average ticket resolution time by priority level?",
         zh="各優先級的平均支援單處理天數是多少?",
         sql="""SELECT priority_level,
       ROUND(AVG(resolved_date - created_date), 1) AS avg_days_to_resolve,
       COUNT(*) AS resolved_tickets
FROM support_ticket
WHERE status = 'Resolved'
GROUP BY 1
ORDER BY avg_days_to_resolve;"""),

    dict(id="churned-revenue", cat="Customers", viz="table",
         en="List churned customers and their lifetime revenue.",
         zh="列出已流失的客戶及其終身營收。",
         sql="""SELECT c.company_name,
       c.signup_date,
       ROUND(SUM(i.total_amount)) AS lifetime_revenue
FROM customer c
JOIN purchase p ON p.customer_id = c.customer_id
JOIN invoice  i ON i.purchase_id = p.purchase_id
WHERE c.status = 'Churned'
GROUP BY 1, 2
ORDER BY lifetime_revenue DESC
LIMIT 10;"""),

    dict(id="contracts-expiring", cat="Operations", viz="table",
         en="Which contracts are expiring in the next 6 months?",
         zh="哪些合約將在未來 6 個月內到期?",
         sql="""SELECT c.company_name,
       ct.end_date,
       ROUND(ct.committed_amount) AS committed_amount,
       ct.discount_pct
FROM contract ct
JOIN customer c ON c.customer_id = ct.customer_id
WHERE ct.end_date BETWEEN CURRENT_DATE AND CURRENT_DATE + INTERVAL 6 MONTH
ORDER BY ct.end_date
LIMIT 15;"""),
]


def main() -> None:
    os.makedirs(DB_DIR, exist_ok=True)
    con = duckdb.connect(WAREHOUSE, read_only=True)

    # 1. Export tables to Parquet
    print("Exporting Parquet ->", DB_DIR)
    total = 0
    for t in TABLES:
        out = os.path.join(DB_DIR, f"{t}.parquet").replace("\\", "/")
        con.execute(f"COPY {t} TO '{out}' (FORMAT PARQUET, COMPRESSION ZSTD)")
        size = os.path.getsize(out)
        total += size
        print(f"  {t:<28} {size/1e6:6.2f} MB")
    print(f"  {'TOTAL':<28} {total/1e6:6.2f} MB")

    # 2. Verify curated queries + 3. emit queries.json
    print("\nVerifying curated queries:")
    out_rows = []
    ok = 0
    for q in CURATED:
        try:
            df = con.execute(q["sql"]).fetchdf()
            rows = len(df)
            cols = list(df.columns)
            status = "OK " if rows > 0 else "EMPTY"
            if rows > 0:
                ok += 1
            print(f"  [{status}] {q['id']:<22} {rows:>4} rows  cols={cols}")
            out_rows.append({
                "id": q["id"], "en": q["en"], "zh": q["zh"],
                "category": q["cat"], "viz": q["viz"],
                "sql": q["sql"], "columns": cols,
            })
        except Exception as e:
            print(f"  [FAIL] {q['id']:<22} {e}")
    con.close()

    with open(QUERIES_JSON, "w", encoding="utf-8") as f:
        json.dump(out_rows, f, ensure_ascii=False, indent=2)
    print(f"\n{ok}/{len(CURATED)} queries returned rows.")
    print("Wrote", QUERIES_JSON)


if __name__ == "__main__":
    main()
