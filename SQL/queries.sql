-- Basic overview
SELECT COUNT(*) AS total_rows FROM churn_data;

-- Churn rate
SELECT
    churned,
    COUNT(*) AS count
FROM churn_data
GROUP BY churned;

-- Average credit limit by churn status
SELECT
    churned,
    AVG(Credit_Limit) AS avg_credit_limit
FROM churn_data
GROUP BY churned;

-- Churn by gender
SELECT
    Gender,
    churned,
    COUNT(*) AS count
FROM churn_data
GROUP BY Gender, churned;

-- Churn by income category
SELECT
    Income_Category,
    churned,
    COUNT(*) AS count
FROM churn_data
GROUP BY Income_Category, churned
ORDER BY Income_Category;