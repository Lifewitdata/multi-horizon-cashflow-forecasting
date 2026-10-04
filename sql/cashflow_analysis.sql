-- Step 11 — SQL layer: the queries a business user actually runs
-- Tables: cashflow_daily (actuals), forecasts_ml (model predictions),
--         forecasts_uncertainty (p10/p50/p90 by horizon)
-- Tested on SQLite. Run via:  python step11_sql.py

-- Q1. Monthly cash summary: inflow, outflow, net, closing balance
SELECT strftime('%Y-%m', date) AS month,
       ROUND(SUM(cash_inflow_lakh), 1)  AS inflow,
       ROUND(SUM(cash_outflow_lakh), 1) AS outflow,
       ROUND(SUM(net_cashflow_lakh), 1) AS net,
       ROUND(MAX(cash_balance_lakh), 1) AS closing_balance
FROM cashflow_daily
GROUP BY 1
ORDER BY 1;

-- Q2. Forecast accuracy by month (ML model vs actual)
SELECT strftime('%Y-%m', f.date) AS month,
       ROUND(AVG(ABS(f.actual - f.hgb)), 2) AS mae,
       ROUND(AVG(f.actual - f.hgb), 2)      AS bias
FROM forecasts_ml f
GROUP BY 1
ORDER BY 1;

-- Q3. Collection efficiency: month-end window vs normal days
SELECT CASE WHEN is_month_end_window = 1 THEN 'month_end_window' ELSE 'normal_day' END AS day_type,
       ROUND(AVG(cash_inflow_lakh), 1) AS avg_inflow,
       COUNT(*) AS days
FROM cashflow_daily
WHERE is_weekend = 0 AND is_holiday = 0
GROUP BY 1;

-- Q4. Payroll-day outflow burden per month
SELECT strftime('%Y-%m', date) AS month,
       ROUND(SUM(CASE WHEN is_payroll_day = 1 THEN cash_outflow_lakh ELSE 0 END), 1) AS payroll_outflow,
       ROUND(SUM(cash_outflow_lakh), 1) AS total_outflow,
       ROUND(100.0 * SUM(CASE WHEN is_payroll_day = 1 THEN cash_outflow_lakh ELSE 0 END)
             / SUM(cash_outflow_lakh), 1) AS payroll_pct
FROM cashflow_daily
GROUP BY 1
ORDER BY 1;

-- Q5. Cash runway alert: days the balance dipped under Rs 15 cr (1500 lakh)
SELECT date, ROUND(cash_balance_lakh, 1) AS balance
FROM cashflow_daily
WHERE cash_balance_lakh < 1500
ORDER BY date;

-- Q6. Driver check: billed vs inflow by quarter (is the link stable?)
SELECT strftime('%Y', date) || '-Q' || ((CAST(strftime('%m', date) AS INT) + 2) / 3) AS quarter,
       ROUND(AVG(billed_lakh), 1)      AS avg_billed,
       ROUND(AVG(cash_inflow_lakh), 1) AS avg_inflow,
       ROUND(AVG(cash_inflow_lakh) / AVG(billed_lakh) * 30, 2) AS collection_ratio
FROM cashflow_daily
GROUP BY 1
ORDER BY 1;

-- Q7. Uncertainty check: how often did actuals fall outside the 80% interval?
SELECT horizon_days,
       COUNT(*) AS days,
       SUM(CASE WHEN actual < p10 OR actual > p90 THEN 1 ELSE 0 END) AS outside_interval,
       ROUND(100.0 * SUM(CASE WHEN actual < p10 OR actual > p90 THEN 1 ELSE 0 END) / COUNT(*), 1)
         AS pct_outside
FROM forecasts_uncertainty
GROUP BY 1
ORDER BY 1;

-- Q8. Worst forecast misses (for the monthly review meeting)
SELECT date,
       ROUND(actual, 1)  AS actual,
       ROUND(hgb, 1) AS forecast,
       ROUND(actual - hgb, 1) AS error
FROM forecasts_ml
ORDER BY ABS(actual - hgb) DESC
LIMIT 10;
