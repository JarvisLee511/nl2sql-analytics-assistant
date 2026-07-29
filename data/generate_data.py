"""
Synthetic data generator for the AWS Purchase Management warehouse.

Builds a realistic ~1.2M-row analytical warehouse across 13 related tables and
writes it to a DuckDB file (data/warehouse.duckdb) for the NL->SQL assistant.

Realism baked in (so questions actually have interesting answers):
  - Revenue grows YoY (~25%/yr) with monthly seasonality (Q4 heavier)
  - Customer segments (Enterprise/Mid-Market/SMB) spend very differently
  - ~12% of customers churn (purchases/usage stop after a churn date)
  - ~12% of invoices end up Pending/Overdue -> realistic AR aging
  - Support-ticket resolution time depends on priority; some stay Open
  - Accounts span real AWS regions across NA / EU / APAC

Deterministic: fixed seed -> identical warehouse every run.
Vectorized with numpy/pandas; Faker only used for the few small name tables.
"""

from __future__ import annotations
import os
from datetime import date, timedelta

import numpy as np
import pandas as pd
import duckdb

SEED = 42
rng = np.random.default_rng(SEED)

# ---- lightweight, dependency-free name pools (fully reproducible) ----
_ADJ = ["Apex", "Blue", "Cedar", "Delta", "Evergreen", "Falcon", "Granite", "Horizon",
        "Ion", "Juniper", "Keystone", "Lumen", "Meridian", "Northwind", "Oasis", "Pinnacle",
        "Quantum", "Redwood", "Summit", "Titan", "Vertex", "Vanguard", "Wavelength", "Zenith",
        "Atlas", "Beacon", "Crest", "Dynamo", "Echo", "Forge"]
_NOUN = ["Systems", "Analytics", "Labs", "Technologies", "Solutions", "Digital", "Networks",
         "Cloud", "Data", "Software", "Logic", "Dynamics", "Ventures", "Industries", "Group",
         "Works", "Robotics", "Media", "Capital", "Health"]
_SUFFIX = ["Inc.", "LLC", "Corp.", "Co.", "Ltd."]
_STREETS = ["Main St", "Oak Ave", "Maple Dr", "Market St", "Park Blvd", "Cedar Ln",
            "Second Ave", "Tech Way", "Innovation Dr", "Commerce St"]
_CITIES = [("Austin", "TX", "73301"), ("Seattle", "WA", "98101"), ("Boston", "MA", "02108"),
           ("Denver", "CO", "80202"), ("Chicago", "IL", "60601"), ("Atlanta", "GA", "30301"),
           ("San Jose", "CA", "95101"), ("New York", "NY", "10001"), ("Portland", "OR", "97201"),
           ("Miami", "FL", "33101")]
_FIRST = ["James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael", "Linda",
          "David", "Elizabeth", "William", "Barbara", "Richard", "Susan", "Joseph", "Jessica",
          "Thomas", "Karen", "Chris", "Nancy", "Daniel", "Lisa", "Kevin", "Amy", "Brian",
          "Angela", "Wei", "Ling", "Hiroshi", "Priya"]
_LAST = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
         "Rodriguez", "Martinez", "Lee", "Chen", "Wang", "Patel", "Kim", "Nguyen",
         "Anderson", "Thomas", "Taylor", "Moore"]

HERE = os.path.dirname(os.path.abspath(__file__))
DUCKDB_PATH = os.path.join(HERE, "warehouse.duckdb")

# ----- scale knobs -----
N_CUSTOMERS = 300
N_EMPLOYEES = 40
N_ACCOUNTS = 600
N_PURCHASES_RAW = 135_000   # filtered down to ~100k by active-window constraints
N_USAGE_RAW = 1_420_000     # filtered down to ~900k
N_TICKETS_RAW = 26_000      # filtered to ~20k

START = date(2022, 1, 1)
END = date(2026, 6, 30)
SPAN_DAYS = (END - START).days

# =====================================================================
# Reference / lookup data
# =====================================================================
INDUSTRIES = ["Technology", "Finance", "Retail", "Healthcare",
              "Government", "Manufacturing", "Media", "Education"]

# segment -> (share, monthly_purchase_rate_weight, spend_multiplier, discount_pct, committed_base)
SEGMENTS = {
    "Enterprise":  dict(share=0.15, weight=6.0, spend=4.0, discount=18.0, committed=1_200_000),
    "Mid-Market":  dict(share=0.35, weight=2.5, spend=1.6, discount=8.0,  committed=300_000),
    "SMB":         dict(share=0.50, weight=1.0, spend=1.0, discount=2.0,  committed=40_000),
}

# (service_name, product_code, category, tier, base_unit_price)
SERVICES = [
    ("Amazon EC2",            "AmazonEC2",      "Compute",    "Compute",    0.096),
    ("AWS Lambda",            "AWSLambda",      "Compute",    "Compute",    0.000017),
    ("Amazon ECS",            "AmazonECS",      "Compute",    "Compute",    0.040),
    ("AWS Fargate",           "AWSFargate",     "Compute",    "Compute",    0.040),
    ("Amazon EC2 Spot",       "AmazonEC2Spot",  "Compute",    "Compute",    0.035),
    ("Amazon S3",             "AmazonS3",       "Storage",    "Storage",    0.023),
    ("Amazon S3 Glacier",     "AmazonGlacier",  "Storage",    "Storage",    0.004),
    ("Amazon EBS",            "AmazonEBS",      "Storage",    "Storage",    0.10),
    ("Amazon EFS",            "AmazonEFS",      "Storage",    "Storage",    0.30),
    ("Amazon RDS",            "AmazonRDS",      "Database",   "Database",   0.17),
    ("Amazon Aurora",         "AmazonAurora",   "Database",   "Database",   0.29),
    ("Amazon DynamoDB",       "AmazonDynamoDB", "Database",   "Database",   0.25),
    ("Amazon Redshift",       "AmazonRedshift", "Database",   "Analytics",  0.25),
    ("Amazon ElastiCache",    "AmazonElastiCache","Database", "Database",   0.068),
    ("Amazon Athena",         "AmazonAthena",   "Analytics",  "Analytics",  5.00),
    ("AWS Glue",              "AWSGlue",        "Analytics",  "Analytics",  0.44),
    ("Amazon EMR",            "AmazonEMR",      "Analytics",  "Analytics",  0.27),
    ("Amazon Kinesis",        "AmazonKinesis",  "Analytics",  "Analytics",  0.015),
    ("Amazon QuickSight",     "AmazonQuickSight","Analytics", "Analytics",  0.30),
    ("Amazon SageMaker",      "AmazonSageMaker","Machine Learning","AI-ML", 1.125),
    ("Amazon Bedrock",        "AmazonBedrock",  "Machine Learning","AI-ML", 0.80),
    ("Amazon Rekognition",    "AmazonRekognition","Machine Learning","AI-ML",1.00),
    ("Amazon Comprehend",     "AmazonComprehend","Machine Learning","AI-ML",0.50),
    ("Amazon CloudFront",     "AmazonCloudFront","Networking","Networking", 0.085),
    ("Elastic Load Balancing","AWSELB",         "Networking", "Networking", 0.0225),
    ("Amazon Route 53",       "AmazonRoute53",  "Networking", "Networking", 0.40),
    ("AWS Direct Connect",    "AWSDirectConnect","Networking","Networking", 0.30),
    ("AWS Key Management",    "AWSKMS",         "Security",   "Security",   0.03),
    ("Amazon GuardDuty",      "AmazonGuardDuty","Security",   "Security",   1.00),
    ("AWS WAF",               "AWSWAF",         "Security",   "Security",   0.60),
]

# (region_name, country, continent)
REGIONS = [
    ("us-east-1", "United States", "North America"),
    ("us-east-2", "United States", "North America"),
    ("us-west-1", "United States", "North America"),
    ("us-west-2", "United States", "North America"),
    ("ca-central-1", "Canada", "North America"),
    ("eu-west-1", "Ireland", "Europe"),
    ("eu-west-2", "United Kingdom", "Europe"),
    ("eu-central-1", "Germany", "Europe"),
    ("eu-north-1", "Sweden", "Europe"),
    ("ap-southeast-1", "Singapore", "Asia Pacific"),
    ("ap-southeast-2", "Australia", "Asia Pacific"),
    ("ap-northeast-1", "Japan", "Asia Pacific"),
    ("ap-south-1", "India", "Asia Pacific"),
    ("sa-east-1", "Brazil", "South America"),
    ("af-south-1", "South Africa", "Africa"),
]

PAYMENT_METHODS = ["Credit Card", "Bank Transfer", "Wire Transfer", "ACH", "PayPal"]
# mean payment delay (days) by method
PAY_DELAY = {"Credit Card": 5, "Bank Transfer": 18, "Wire Transfer": 22, "ACH": 12, "PayPal": 3}

ISSUE_TYPES = ["Billing Inquiry", "Technical Support", "Service Outage",
               "Access / IAM", "Quota Increase", "Performance", "Security Concern"]
PRIORITIES = ["Low", "Medium", "High", "Critical"]
# resolution time (days) mean by priority
RESOLVE_DAYS = {"Low": 9, "Medium": 5, "High": 2, "Critical": 1}


def to_dt(days_from_start):
    """Vectorized int-days-offset -> pandas datetime."""
    base = np.datetime64(START)
    return pd.to_datetime(base + days_from_start.astype("timedelta64[D]"))


def time_weights(n_days):
    """Per-day sampling weight = YoY growth * monthly seasonality."""
    days = np.arange(n_days)
    yrs = days / 365.0
    growth = (1.25) ** yrs                      # ~25% YoY growth
    months = (np.datetime64(START) + days.astype("timedelta64[D]")).astype("datetime64[M]")
    month_num = months.astype(int) % 12 + 1     # 1..12
    season = np.array([0.85, 0.85, 0.95, 1.0, 1.0, 1.05,
                       1.0, 1.0, 1.1, 1.15, 1.25, 1.2])  # Q4 heavier
    w = growth * season[month_num - 1]
    return w / w.sum()


# =====================================================================
# 1. customer
# =====================================================================
seg_names = list(SEGMENTS)
seg_p = np.array([SEGMENTS[s]["share"] for s in seg_names])
cust_segment = rng.choice(seg_names, size=N_CUSTOMERS, p=seg_p)

signup_offset = rng.integers(- 365, SPAN_DAYS - 120, size=N_CUSTOMERS)  # some signed up before START
signup_date = pd.to_datetime(np.datetime64(START) + signup_offset.astype("timedelta64[D]"))

is_churned = rng.random(N_CUSTOMERS) < 0.12
# churn date: somewhere between signup+180d and END
churn_offset = signup_offset + 180 + rng.integers(0, 900, size=N_CUSTOMERS)
churn_offset = np.minimum(churn_offset, SPAN_DAYS)
churn_dt = np.where(is_churned, np.datetime64(START) + churn_offset.astype("timedelta64[D]"),
                    np.datetime64("2100-01-01"))
churn_dt = pd.to_datetime(churn_dt)

# unique company names from the 30x20 = 600-combo pool
_base = [f"{a} {n}" for a in _ADJ for n in _NOUN]
rng.shuffle(_base)
_base = _base[:N_CUSTOMERS]
company_name = [f"{b} {_SUFFIX[rng.integers(len(_SUFFIX))]}" for b in _base]
_slug = [b.lower().replace(" ", "") for b in _base]
_ci = rng.integers(0, len(_CITIES), N_CUSTOMERS)
billing_address = [f"{rng.integers(100, 9999)} {_STREETS[rng.integers(len(_STREETS))]}, "
                   f"{_CITIES[c][0]}, {_CITIES[c][1]} {_CITIES[c][2]}" for c in _ci]
tax_id = [f"{chr(65 + rng.integers(26))}{chr(65 + rng.integers(26))}-{rng.integers(10_000_000, 99_999_999)}"
          for _ in range(N_CUSTOMERS)]
phone_number = [f"({rng.integers(200, 999)}) {rng.integers(200, 999)}-{rng.integers(1000, 9999)}"
                for _ in range(N_CUSTOMERS)]

customer = pd.DataFrame({
    "customer_id": np.arange(1, N_CUSTOMERS + 1),
    "company_name": company_name,
    "industry": rng.choice(INDUSTRIES, size=N_CUSTOMERS),
    "segment": cust_segment,
    "billing_address": billing_address,
    "email": [f"contact@{s}.com" for s in _slug],
    "tax_id": tax_id,
    "phone_number": phone_number,
    "signup_date": signup_date,
    "status": np.where(is_churned, "Churned", "Active"),
})

# =====================================================================
# 2. employee
# =====================================================================
depts = rng.choice(["Sales", "Support", "Account Management"], size=N_EMPLOYEES, p=[0.3, 0.4, 0.3])
emp_name = [f"{_FIRST[rng.integers(len(_FIRST))]} {_LAST[rng.integers(len(_LAST))]}"
            for _ in range(N_EMPLOYEES)]
emp_email = [f"{n.lower().replace(' ', '.')}.{i}@awspartner.com" for i, n in enumerate(emp_name)]
employee = pd.DataFrame({
    "employee_id": np.arange(1, N_EMPLOYEES + 1),
    "employee_name": emp_name,
    "department": depts,
    "team": rng.choice(["NA Team", "EMEA Team", "APAC Team"], size=N_EMPLOYEES),
    "email": emp_email,
    "hire_date": to_dt(rng.integers(-1500, SPAN_DAYS - 200, size=N_EMPLOYEES)),
})
am_ids = employee.loc[employee.department == "Account Management", "employee_id"].to_numpy()
if len(am_ids) == 0:
    am_ids = employee.employee_id.to_numpy()[:5]

# =====================================================================
# 3. aws_service
# =====================================================================
aws_service = pd.DataFrame(SERVICES, columns=[
    "service_name", "product_code", "category", "service_tier", "current_unit_price"])
aws_service.insert(0, "service_id", np.arange(1, len(aws_service) + 1))
N_SERVICES = len(aws_service)
svc_price = aws_service.set_index("service_id")["current_unit_price"].to_dict()

# =====================================================================
# 11. region  +  2. aws_account  +  12. account_region
# =====================================================================
region = pd.DataFrame(REGIONS, columns=["region_name", "country", "continent"])
region.insert(0, "region_id", np.arange(1, len(region) + 1))
N_REGIONS = len(region)

acct_customer = rng.integers(1, N_CUSTOMERS + 1, size=N_ACCOUNTS)
aws_account = pd.DataFrame({
    "account_id": np.arange(1, N_ACCOUNTS + 1),
    "account_name": [f"{customer.loc[c-1,'company_name'][:18].strip()}-{rng.integers(100,999)}"
                     for c in acct_customer],
    "customer_id": acct_customer,
})
acct_to_cust = dict(zip(aws_account.account_id, aws_account.customer_id))

ar_rows = []
for aid in aws_account.account_id:
    k = rng.integers(1, 4)
    for rid in rng.choice(np.arange(1, N_REGIONS + 1), size=k, replace=False):
        ar_rows.append((int(aid), int(rid)))
account_region = pd.DataFrame(ar_rows, columns=["account_id", "region_id"])

# =====================================================================
# 13. contract  (one per customer)
# =====================================================================
contract = pd.DataFrame({
    "contract_id": np.arange(1, N_CUSTOMERS + 1),
    "customer_id": customer.customer_id,
    "start_date": customer.signup_date,
    "end_date": customer.signup_date + pd.to_timedelta(365 * rng.integers(1, 4, N_CUSTOMERS), unit="D"),
    "committed_amount": [round(SEGMENTS[s]["committed"] * rng.uniform(0.6, 1.6), 2) for s in cust_segment],
    "discount_pct": [SEGMENTS[s]["discount"] for s in cust_segment],
})

# =====================================================================
# 9. account_manager_assignment (one AM per customer)
# =====================================================================
account_manager_assignment = pd.DataFrame({
    "assignment_id": np.arange(1, N_CUSTOMERS + 1),
    "customer_id": customer.customer_id,
    "employee_id": rng.choice(am_ids, size=N_CUSTOMERS),
    "assignment_date": customer.signup_date + pd.to_timedelta(rng.integers(0, 60, N_CUSTOMERS), unit="D"),
})

# helper: per-customer active window as int day-offsets from START
cust_signoff = signup_offset.copy()
cust_churnoff = np.where(is_churned, churn_offset, SPAN_DAYS)
seg_weight = np.array([SEGMENTS[s]["weight"] for s in cust_segment])
seg_spend = np.array([SEGMENTS[s]["spend"] for s in cust_segment])
cust_discount = np.array([SEGMENTS[s]["discount"] for s in cust_segment])
tw = time_weights(SPAN_DAYS + 1)

# =====================================================================
# 4. purchase  (oversample then filter by each customer's active window)
# =====================================================================
cust_p = seg_weight / seg_weight.sum()
p_cust = rng.choice(np.arange(N_CUSTOMERS), size=N_PURCHASES_RAW, p=cust_p)
p_day = rng.choice(np.arange(SPAN_DAYS + 1), size=N_PURCHASES_RAW, p=tw)
# keep only purchases inside the customer's active window
mask = (p_day >= np.maximum(cust_signoff[p_cust], 0)) & (p_day <= cust_churnoff[p_cust])
p_cust, p_day = p_cust[mask], p_day[mask]
N_PURCHASE = len(p_cust)

p_service = rng.integers(1, N_SERVICES + 1, size=N_PURCHASE)
base_price = aws_service.set_index("service_id")["current_unit_price"].to_numpy()
p_unit = base_price[p_service - 1] * rng.uniform(0.9, 1.15, N_PURCHASE)
p_unit = p_unit * (1 - cust_discount[p_cust] / 100.0)          # contract discount
p_qty = (rng.gamma(2.0, 30.0, N_PURCHASE) * seg_spend[p_cust]).astype(int) + 1

purchase = pd.DataFrame({
    "purchase_id": np.arange(1, N_PURCHASE + 1),
    "purchase_date": to_dt(p_day),
    "quantity": p_qty,
    "unit_price_at_purchase": np.round(p_unit, 4),
    "customer_id": p_cust + 1,
    "service_id": p_service,
})

# =====================================================================
# 5. invoice  +  6. payment   (drives AR aging)
# =====================================================================
inv_date_off = p_day + rng.integers(0, 4, N_PURCHASE)
total_amount = np.round(p_qty * p_unit, 2)
method = rng.choice(PAYMENT_METHODS, size=N_PURCHASE, p=[0.4, 0.2, 0.1, 0.2, 0.1])
delay = np.array([PAY_DELAY[m] for m in method]) + rng.integers(-2, 10, N_PURCHASE)
delay = np.clip(delay, 0, None)
pay_day = inv_date_off + delay
will_pay = rng.random(N_PURCHASE) < 0.90
paid_in_window = will_pay & (pay_day <= SPAN_DAYS)

status = np.full(N_PURCHASE, "Pending", dtype=object)
status[paid_in_window] = "Paid"
overdue = (~will_pay) & ((SPAN_DAYS - inv_date_off) > 45)
status[overdue] = "Overdue"

invoice = pd.DataFrame({
    "invoice_id": np.arange(1, N_PURCHASE + 1),
    "purchase_id": purchase.purchase_id,
    "invoice_date": to_dt(inv_date_off),
    "total_amount": total_amount,
    "payment_status": status,
})

pay_idx = np.where(paid_in_window)[0]
payment = pd.DataFrame({
    "payment_id": np.arange(1, len(pay_idx) + 1),
    "invoice_id": invoice.invoice_id.to_numpy()[pay_idx],
    "payment_date": to_dt(pay_day[pay_idx]),
    "payment_method": method[pay_idx],
    "amount_paid": total_amount[pay_idx],
})

# =====================================================================
# 10. usage_log  (the big table; oversample then filter by active window)
# =====================================================================
acct_idx = rng.integers(0, N_ACCOUNTS, size=N_USAGE_RAW)
u_cust0 = acct_customer[acct_idx] - 1                       # 0-based customer
u_day = rng.choice(np.arange(SPAN_DAYS + 1), size=N_USAGE_RAW, p=tw)
umask = (u_day >= np.maximum(cust_signoff[u_cust0], 0)) & (u_day <= cust_churnoff[u_cust0])
acct_idx, u_day = acct_idx[umask], u_day[umask]
N_USAGE = len(acct_idx)

u_service = rng.integers(1, N_SERVICES + 1, size=N_USAGE)
u_hours = (rng.gamma(2.0, 40.0, N_USAGE)).astype(int) + 1
u_cost = np.round(u_hours * base_price[u_service - 1] * rng.uniform(0.8, 1.3, N_USAGE), 2)
usage_log = pd.DataFrame({
    "usage_id": np.arange(1, N_USAGE + 1),
    "account_id": acct_idx + 1,
    "service_id": u_service,
    "usage_date": to_dt(u_day),
    "usage_hours": u_hours,
    "cost": u_cost,
})

# =====================================================================
# 7. support_ticket  (volume weighted to active customers; resolution SLA)
# =====================================================================
t_cust = rng.choice(np.arange(N_CUSTOMERS), size=N_TICKETS_RAW, p=cust_p)
t_day = rng.choice(np.arange(SPAN_DAYS + 1), size=N_TICKETS_RAW, p=tw)
tmask = (t_day >= np.maximum(cust_signoff[t_cust], 0)) & (t_day <= cust_churnoff[t_cust])
t_cust, t_day = t_cust[tmask], t_day[tmask]
N_TICKET = len(t_cust)

t_priority = rng.choice(PRIORITIES, size=N_TICKET, p=[0.35, 0.4, 0.2, 0.05])
res_mean = np.array([RESOLVE_DAYS[p] for p in t_priority])
res_days = np.maximum(0, (rng.gamma(2.0, res_mean / 2.0)).astype(int))
resolved_off = t_day + res_days
still_open = (rng.random(N_TICKET) < 0.15) | (resolved_off > SPAN_DAYS)
t_status = np.where(still_open, "Open", "Resolved")
resolved_dt = np.where(still_open, np.datetime64("NaT"),
                       np.datetime64(START) + resolved_off.astype("timedelta64[D]"))

support_ticket = pd.DataFrame({
    "ticket_id": np.arange(1, N_TICKET + 1),
    "customer_id": t_cust + 1,
    "issue_type": rng.choice(ISSUE_TYPES, size=N_TICKET),
    "priority_level": t_priority,
    "created_date": to_dt(t_day),
    "resolved_date": pd.to_datetime(resolved_dt),
    "status": t_status,
})

# =====================================================================
# Write to DuckDB
# =====================================================================
DDL = """
DROP TABLE IF EXISTS account_region; DROP TABLE IF EXISTS usage_log;
DROP TABLE IF EXISTS account_manager_assignment; DROP TABLE IF EXISTS payment;
DROP TABLE IF EXISTS invoice; DROP TABLE IF EXISTS support_ticket;
DROP TABLE IF EXISTS contract; DROP TABLE IF EXISTS region;
DROP TABLE IF EXISTS employee; DROP TABLE IF EXISTS purchase;
DROP TABLE IF EXISTS aws_account; DROP TABLE IF EXISTS aws_service;
DROP TABLE IF EXISTS customer;

CREATE TABLE customer (
  customer_id INTEGER PRIMARY KEY, company_name VARCHAR, industry VARCHAR,
  segment VARCHAR, billing_address VARCHAR, email VARCHAR, tax_id VARCHAR,
  phone_number VARCHAR, signup_date DATE, status VARCHAR);
CREATE TABLE employee (
  employee_id INTEGER PRIMARY KEY, employee_name VARCHAR, department VARCHAR,
  team VARCHAR, email VARCHAR, hire_date DATE);
CREATE TABLE aws_service (
  service_id INTEGER PRIMARY KEY, service_name VARCHAR, product_code VARCHAR,
  category VARCHAR, service_tier VARCHAR, current_unit_price DECIMAL(12,6));
CREATE TABLE aws_account (
  account_id INTEGER PRIMARY KEY, account_name VARCHAR, customer_id INTEGER);
CREATE TABLE region (
  region_id INTEGER PRIMARY KEY, region_name VARCHAR, country VARCHAR, continent VARCHAR);
CREATE TABLE account_region (account_id INTEGER, region_id INTEGER);
CREATE TABLE contract (
  contract_id INTEGER PRIMARY KEY, customer_id INTEGER, start_date DATE,
  end_date DATE, committed_amount DECIMAL(14,2), discount_pct DECIMAL(5,2));
CREATE TABLE account_manager_assignment (
  assignment_id INTEGER PRIMARY KEY, customer_id INTEGER, employee_id INTEGER,
  assignment_date DATE);
CREATE TABLE purchase (
  purchase_id INTEGER PRIMARY KEY, purchase_date DATE, quantity INTEGER,
  unit_price_at_purchase DECIMAL(12,4), customer_id INTEGER, service_id INTEGER);
CREATE TABLE invoice (
  invoice_id INTEGER PRIMARY KEY, purchase_id INTEGER, invoice_date DATE,
  total_amount DECIMAL(14,2), payment_status VARCHAR);
CREATE TABLE payment (
  payment_id INTEGER PRIMARY KEY, invoice_id INTEGER, payment_date DATE,
  payment_method VARCHAR, amount_paid DECIMAL(14,2));
CREATE TABLE support_ticket (
  ticket_id INTEGER PRIMARY KEY, customer_id INTEGER, issue_type VARCHAR,
  priority_level VARCHAR, created_date DATE, resolved_date DATE, status VARCHAR);
CREATE TABLE usage_log (
  usage_id INTEGER PRIMARY KEY, account_id INTEGER, service_id INTEGER,
  usage_date DATE, usage_hours INTEGER, cost DECIMAL(14,2));
"""

TABLES = {
    "customer": customer, "employee": employee, "aws_service": aws_service,
    "aws_account": aws_account, "region": region, "account_region": account_region,
    "contract": contract, "account_manager_assignment": account_manager_assignment,
    "purchase": purchase, "invoice": invoice, "payment": payment,
    "support_ticket": support_ticket, "usage_log": usage_log,
}

if os.path.exists(DUCKDB_PATH):
    os.remove(DUCKDB_PATH)
con = duckdb.connect(DUCKDB_PATH)
con.execute(DDL)
total = 0
for name, df in TABLES.items():
    con.register("df_tmp", df)
    con.execute(f"INSERT INTO {name} SELECT * FROM df_tmp")
    con.unregister("df_tmp")
    total += len(df)
    print(f"  {name:<28} {len(df):>9,} rows")
print(f"  {'TOTAL':<28} {total:>9,} rows")

# ----- sanity checks -----
print("\nSanity checks:")
print(con.execute("""
  SELECT c.segment, COUNT(DISTINCT c.customer_id) customers,
         ROUND(SUM(i.total_amount)) revenue
  FROM customer c JOIN purchase p ON p.customer_id=c.customer_id
  JOIN invoice i ON i.purchase_id=p.purchase_id
  GROUP BY 1 ORDER BY revenue DESC
""").fetchdf().to_string(index=False))
print()
print(con.execute("""
  SELECT payment_status, COUNT(*) invoices,
         ROUND(100.0*COUNT(*)/SUM(COUNT(*)) OVER (),1) pct
  FROM invoice GROUP BY 1 ORDER BY invoices DESC
""").fetchdf().to_string(index=False))
con.close()
print(f"\nWrote {DUCKDB_PATH} ({os.path.getsize(DUCKDB_PATH)/1e6:.1f} MB)")
