"""
NL->SQL benchmark: valid-SQL rate + LLM-judged correctness.

Exact result-set matching is the wrong metric for open-ended business questions:
two correct queries can differ in columns, rounding, revenue definition, or sort
order. So we measure:

  1. Valid-SQL rate  — does the generated SQL execute against the warehouse?
  2. Correctness     — an independent judge model (Opus 4.8) decides whether the
                       generated query correctly answers the question, given the
                       hand-verified gold SQL and a preview of both result sets.
                       Reasonable variation is accepted.

Usage:
    pip install anthropic duckdb
    export ANTHROPIC_API_KEY=sk-ant-...    (or set on Windows)
    python eval/run_eval.py            # -> eval/results.md
"""

from __future__ import annotations
import json
import os
from datetime import date
import duckdb
from anthropic import Anthropic

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WAREHOUSE = os.path.join(ROOT, "data", "warehouse.duckdb")
QUERIES = os.path.join(ROOT, "queries.json")
RESULTS = os.path.join(HERE, "results.md")

MODELS = ["claude-opus-4-8", "claude-sonnet-4-6", "claude-haiku-4-5"]
JUDGE = "claude-opus-4-8"

SCHEMA_DDL = """customer(customer_id, company_name, industry, segment, signup_date, status)
aws_account(account_id, account_name, customer_id -> customer)
aws_service(service_id, service_name, product_code, category, service_tier, current_unit_price)
purchase(purchase_id, purchase_date, quantity, unit_price_at_purchase, customer_id, service_id)
invoice(invoice_id, purchase_id -> purchase, invoice_date, total_amount, payment_status)
payment(payment_id, invoice_id -> invoice, payment_date, payment_method, amount_paid)
support_ticket(ticket_id, customer_id, issue_type, priority_level, created_date, resolved_date, status)
employee(employee_id, employee_name, department, team, hire_date)
account_manager_assignment(assignment_id, customer_id, employee_id -> employee, assignment_date)
usage_log(usage_id, account_id -> aws_account, service_id -> aws_service, usage_date, usage_hours, cost)
region(region_id, region_name, country, continent)
account_region(account_id -> aws_account, region_id -> region)
contract(contract_id, customer_id, start_date, end_date, committed_amount, discount_pct)"""

GEN_SYSTEM = f"""You write DuckDB SQL for this AWS-reseller warehouse. Return ONE SELECT
statement that answers the question. Return ONLY SQL, no prose, no markdown fences.

Schema:
{SCHEMA_DDL}"""

JUDGE_SYSTEM = """You grade whether a CANDIDATE SQL query correctly answers a business
question, using a hand-verified GOLD query as reference. Accept reasonable variation:
different but valid column sets, rounding, sort order, or an equally valid metric
definition (e.g. revenue from invoice totals vs quantity*price). Mark INCORRECT only
if the candidate answers a different question, is missing the core answer, or is wrong.
Reply with exactly one word on the first line: CORRECT or INCORRECT."""


def clean_sql(text: str) -> str:
    s = text.strip()
    if s.startswith("```"):
        parts = s.split("```")
        s = parts[1] if len(parts) >= 2 else s.strip("`")
        if s.lower().startswith("sql"):
            s = s.split("\n", 1)[1] if "\n" in s else s[3:]
    return s.strip().rstrip(";").strip()


def preview(con, sql, n=8):
    cur = con.execute(sql)
    cols = [d[0] for d in cur.description]
    rows = cur.fetchmany(n)
    return cols, [tuple(str(x) for x in r) for r in rows]


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY first.")
    client = Anthropic()
    con = duckdb.connect(WAREHOUSE, read_only=True)
    cases = json.load(open(QUERIES, encoding="utf-8"))

    valid = {m: 0 for m in MODELS}
    correct = {m: 0 for m in MODELS}
    rows_md = []

    for case in cases:
        q, gold_sql = case["en"], case["sql"]
        gcols, gprev = preview(con, gold_sql)
        line = {"id": case["id"]}
        for model in MODELS:
            ok_valid = ok_correct = False
            try:
                resp = client.messages.create(model=model, max_tokens=2000,
                    system=GEN_SYSTEM, messages=[{"role": "user", "content": q}])
                gen_sql = clean_sql("".join(b.text for b in resp.content if b.type == "text"))
                ccols, cprev = preview(con, gen_sql)   # raises if invalid
                ok_valid = True
                jmsg = (f"Question: {q}\n\nGOLD SQL:\n{gold_sql}\nGOLD result ({gcols}):\n{gprev}\n\n"
                        f"CANDIDATE SQL:\n{gen_sql}\nCANDIDATE result ({ccols}):\n{cprev}")
                jr = client.messages.create(model=JUDGE, max_tokens=20,
                    system=JUDGE_SYSTEM, messages=[{"role": "user", "content": jmsg}])
                verdict = "".join(b.text for b in jr.content if b.type == "text").strip().upper()
                ok_correct = verdict.startswith("CORRECT")
            except Exception:
                pass
            valid[model] += ok_valid
            correct[model] += ok_correct
            line[model] = (ok_valid, ok_correct)
        print(f"  {case['id']:<22} " +
              "  ".join(f"{m.split('-')[1]}:{'C' if line[m][1] else ('v' if line[m][0] else 'x')}" for m in MODELS))
        rows_md.append(line)

    n = len(cases)
    with open(RESULTS, "w", encoding="utf-8") as f:
        f.write("# NL→SQL benchmark\n\n")
        f.write(f"{n} natural-language questions → SQL, generated by each model and executed against the "
                f"1.16M-row warehouse. **Correctness is judged by an independent model ({JUDGE})** "
                "against hand-verified gold SQL, accepting reasonable variation (columns, rounding, sort, metric definition).\n\n")
        f.write(f"> **Run {date.today():%Y-%m-%d}.** These were the current Claude models on that date. "
                "Scores are point-in-time — re-running against newer models will produce different numbers.\n\n")
        f.write("| Model | Valid SQL | Correct (LLM-judged) |\n|---|---|---|\n")
        for m in MODELS:
            f.write(f"| `{m}` | {valid[m]}/{n} ({100*valid[m]/n:.0f}%) | **{correct[m]}/{n} ({100*correct[m]/n:.0f}%)** |\n")
        f.write("\n<details><summary>Per-question (C = correct · v = valid but judged wrong · x = invalid SQL)</summary>\n\n")
        f.write("| Question | " + " | ".join(m.split("-")[1] for m in MODELS) + " |\n")
        f.write("|---" * (len(MODELS) + 1) + "|\n")
        for r in rows_md:
            cells = ["C" if r[m][1] else ("v" if r[m][0] else "x") for m in MODELS]
            f.write(f"| {r['id']} | " + " | ".join(cells) + " |\n")
        f.write("\n</details>\n")
    con.close()
    print("\nValid:   " + "  ".join(f"{m.split('-')[1]} {valid[m]}/{n}" for m in MODELS))
    print("Correct: " + "  ".join(f"{m.split('-')[1]} {correct[m]}/{n}" for m in MODELS))
    print("Wrote", RESULTS)


if __name__ == "__main__":
    main()
