-- ============================================================================
-- GOOGLE SUSTAINABILITY ANALYTICS - SQL QUERY LIBRARY
-- Complex queries demonstrating SQL proficiency for TPM role
-- ============================================================================

-- ============================================================================
-- 1. MONTHLY EMISSIONS TREND WITH MOVING AVERAGES
-- Demonstrates: Window functions, aggregation, date formatting
-- ============================================================================
SELECT 
    strftime('%Y-%m', date) as month,
    ROUND(SUM(scope1_tonnes), 0) as scope1,
    ROUND(SUM(scope2_market_tonnes), 0) as scope2,
    ROUND(SUM(scope3_tonnes), 0) as scope3,
    ROUND(SUM(scope1_tonnes + scope2_market_tonnes + scope3_tonnes), 0) as total,
    ROUND(AVG(SUM(scope1_tonnes + scope2_market_tonnes + scope3_tonnes)) 
          OVER (ORDER BY date ROWS BETWEEN 2 PRECEDING AND CURRENT ROW), 0) as moving_avg_3m,
    ROUND(AVG(SUM(scope1_tonnes + scope2_market_tonnes + scope3_tonnes)) 
          OVER (ORDER BY date ROWS BETWEEN 5 PRECEDING AND CURRENT ROW), 0) as moving_avg_6m
FROM emissions_monthly
GROUP BY strftime('%Y-%m', date)
ORDER BY month DESC
LIMIT 24;

-- ============================================================================
-- 2. YEAR-OVER-YEAR COMPARISON WITH LAG
-- Demonstrates: LAG window function, percentage calculations, CTEs
-- ============================================================================
WITH monthly_totals AS (
    SELECT 
        strftime('%Y', date) as year,
        strftime('%m', date) as month,
        SUM(scope1_tonnes + scope2_market_tonnes + scope3_tonnes) as total_emissions
    FROM emissions_monthly
    GROUP BY year, month
)
SELECT 
    year,
    month,
    ROUND(total_emissions, 0) as current_emissions,
    ROUND(LAG(total_emissions, 12) OVER (ORDER BY year, month), 0) as prior_year_emissions,
    ROUND(total_emissions - LAG(total_emissions, 12) OVER (ORDER BY year, month), 0) as absolute_change,
    ROUND((total_emissions - LAG(total_emissions, 12) OVER (ORDER BY year, month)) / 
          LAG(total_emissions, 12) OVER (ORDER BY year, month) * 100, 1) as pct_change
FROM monthly_totals
WHERE year >= '2023'
ORDER BY year, month;

-- ============================================================================
-- 3. SCOPE 3 CATEGORY ANALYSIS WITH PARETO
-- Demonstrates: CROSS JOIN, window functions (RANK, cumulative SUM)
-- ============================================================================
WITH scope3_totals AS (
    SELECT 
        c.category_id,
        c.category_name,
        c.description,
        ROUND(SUM(e.scope3_tonnes) * (c.typical_pct_of_scope3 / 100), 0) as estimated_tonnes
    FROM emissions_monthly e
    CROSS JOIN scope3_categories c
    WHERE strftime('%Y', e.date) = '2023'
    GROUP BY c.category_id
)
SELECT 
    category_id,
    category_name,
    description,
    estimated_tonnes,
    RANK() OVER (ORDER BY estimated_tonnes DESC) as rank,
    ROUND(estimated_tonnes * 100.0 / SUM(estimated_tonnes) OVER (), 1) as pct_of_scope3,
    ROUND(SUM(estimated_tonnes) OVER (ORDER BY estimated_tonnes DESC 
          ROWS UNBOUNDED PRECEDING) * 100.0 / 
          SUM(estimated_tonnes) OVER (), 1) as cumulative_pct
FROM scope3_totals
ORDER BY estimated_tonnes DESC;

-- ============================================================================
-- 4. TARGET PROGRESS TRACKING
-- Demonstrates: Multiple CTEs, CROSS JOIN, complex arithmetic
-- ============================================================================
WITH current_emissions AS (
    SELECT 
        strftime('%Y', date) as year,
        SUM(scope1_tonnes + scope2_market_tonnes) as scope12_total,
        SUM(scope3_tonnes) as scope3_total
    FROM emissions_monthly
    GROUP BY year
),
target_info AS (
    SELECT 
        scope,
        baseline_year,
        baseline_emissions,
        target_year,
        target_emissions,
        reduction_percent
    FROM emission_targets
)
SELECT 
    ce.year,
    ROUND(ce.scope12_total, 0) as scope12_actual,
    ROUND(ce.scope3_total, 0) as scope3_actual,
    ROUND(ce.scope12_total + ce.scope3_total, 0) as total_actual,
    ROUND(ti_scope12.baseline_emissions, 0) as scope12_baseline,
    ROUND(ti_scope12.target_emissions, 0) as scope12_target,
    ROUND(ti_scope12.baseline_emissions - 
          ((ti_scope12.baseline_emissions - ti_scope12.target_emissions) * 
           (ce.year - ti_scope12.baseline_year) / 
           (ti_scope12.target_year - ti_scope12.baseline_year)), 0) as scope12_trajectory,
    ROUND((ti_scope12.baseline_emissions - ce.scope12_total) / 
          (ti_scope12.baseline_emissions - ti_scope12.target_emissions) * 100, 1) as scope12_pct_progress,
    CASE 
        WHEN ce.scope12_total <= ti_scope12.baseline_emissions - 
             ((ti_scope12.baseline_emissions - ti_scope12.target_emissions) * 
              (ce.year - ti_scope12.baseline_year) / 
              (ti_scope12.target_year - ti_scope12.baseline_year))
        THEN 'On Track'
        ELSE 'Behind Target'
    END as status
FROM current_emissions ce
CROSS JOIN (SELECT * FROM target_info WHERE scope = 'Scope 1+2') ti_scope12
WHERE ce.year >= ti_scope12.baseline_year
ORDER BY ce.year;

-- ============================================================================
-- 5. FACILITY PERFORMANCE RANKING WITH NTILE
-- Demonstrates: NTILE for quartiles, complex joins, emission intensity
-- ============================================================================
WITH facility_metrics AS (
    SELECT 
        f.facility_id,
        f.facility_name,
        f.region,
        f.facility_type,
        ROUND(SUM(e.scope1_tonnes + e.scope2_market_tonnes + e.scope3_tonnes), 0) as total_emissions,
        ROUND(AVG(e.renewable_pct), 0) as avg_renewable_pct,
        ROUND(SUM(e.scope1_tonnes + e.scope2_market_tonnes + e.scope3_tonnes) / 
              SUM(b.revenue_millions), 1) as emission_intensity,
        ROUND(SUM(b.revenue_millions), 1) as total_revenue
    FROM facilities f
    JOIN emissions_monthly e ON f.facility_id = e.facility_id
    JOIN business_metrics b ON f.facility_id = b.facility_id AND e.date = b.date
    WHERE strftime('%Y', e.date) = '2023'
    GROUP BY f.facility_id
)
SELECT 
    facility_id,
    facility_name,
    region,
    facility_type,
    total_emissions,
    total_revenue,
    emission_intensity,
    avg_renewable_pct,
    RANK() OVER (ORDER BY emission_intensity ASC) as intensity_rank,
    NTILE(4) OVER (ORDER BY emission_intensity ASC) as performance_quartile,
    CASE 
        WHEN NTILE(4) OVER (ORDER BY emission_intensity ASC) = 1 THEN 'Top Performer'
        WHEN NTILE(4) OVER (ORDER BY emission_intensity ASC) = 2 THEN 'Above Average'
        WHEN NTILE(4) OVER (ORDER BY emission_intensity ASC) = 3 THEN 'Below Average'
        ELSE 'Needs Improvement'
    END as performance_category
FROM facility_metrics
ORDER BY emission_intensity ASC;

-- ============================================================================
-- 6. REGIONAL EMISSIONS BREAKDOWN WITH SUBTOTALS
-- Demonstrates: GROUP BY with ROLLUP simulation, UNION ALL
-- ============================================================================
SELECT 
    COALESCE(f.region, 'TOTAL') as region,
    COALESCE(f.facility_type, 'All Types') as facility_type,
    COUNT(DISTINCT f.facility_id) as facility_count,
    ROUND(SUM(e.scope1_tonnes), 0) as scope1_total,
    ROUND(SUM(e.scope2_market_tonnes), 0) as scope2_total,
    ROUND(SUM(e.scope3_tonnes), 0) as scope3_total,
    ROUND(SUM(e.scope1_tonnes + e.scope2_market_tonnes + e.scope3_tonnes), 0) as total_emissions
FROM facilities f
JOIN emissions_monthly e ON f.facility_id = e.facility_id
WHERE strftime('%Y', e.date) = '2023'
GROUP BY f.region, f.facility_type

UNION ALL

SELECT 
    f.region,
    'All Types' as facility_type,
    COUNT(DISTINCT f.facility_id),
    ROUND(SUM(e.scope1_tonnes), 0),
    ROUND(SUM(e.scope2_market_tonnes), 0),
    ROUND(SUM(e.scope3_tonnes), 0),
    ROUND(SUM(e.scope1_tonnes + e.scope2_market_tonnes + e.scope3_tonnes), 0)
FROM facilities f
JOIN emissions_monthly e ON f.facility_id = e.facility_id
WHERE strftime('%Y', e.date) = '2023'
GROUP BY f.region

UNION ALL

SELECT 
    'TOTAL' as region,
    'All Types' as facility_type,
    COUNT(DISTINCT f.facility_id),
    ROUND(SUM(e.scope1_tonnes), 0),
    ROUND(SUM(e.scope2_market_tonnes), 0),
    ROUND(SUM(e.scope3_tonnes), 0),
    ROUND(SUM(e.scope1_tonnes + e.scope2_market_tonnes + e.scope3_tonnes), 0)
FROM facilities f
JOIN emissions_monthly e ON f.facility_id = e.facility_id
WHERE strftime('%Y', e.date) = '2023'

ORDER BY 
    CASE WHEN region = 'TOTAL' THEN 1 ELSE 0 END,
    region,
    CASE WHEN facility_type = 'All Types' THEN 1 ELSE 0 END,
    facility_type;

-- ============================================================================
-- 7. DATA QUALITY AUDIT
-- Demonstrates: CASE aggregation, data validation, percentage calculations
-- ============================================================================
WITH data_quality AS (
    SELECT 
        COUNT(*) as total_records,
        SUM(CASE WHEN scope1_tonnes IS NULL THEN 1 ELSE 0 END) as missing_scope1,
        SUM(CASE WHEN scope2_market_tonnes IS NULL THEN 1 ELSE 0 END) as missing_scope2,
        SUM(CASE WHEN scope3_tonnes IS NULL THEN 1 ELSE 0 END) as missing_scope3,
        SUM(CASE WHEN scope2_market_tonnes > scope2_location_tonnes THEN 1 ELSE 0 END) as inconsistent_scope2,
        SUM(CASE WHEN scope1_tonnes < 0 OR scope2_market_tonnes < 0 OR scope3_tonnes < 0 
                 THEN 1 ELSE 0 END) as negative_values,
        SUM(CASE WHEN renewable_pct < 0 OR renewable_pct > 100 THEN 1 ELSE 0 END) as invalid_renewable
    FROM emissions_monthly
)
SELECT 
    'Total Records' as check_type,
    total_records as count,
    100.0 as pass_rate_pct
FROM data_quality

UNION ALL

SELECT 
    'Complete Scope 1',
    total_records - missing_scope1,
    ROUND((total_records - missing_scope1) * 100.0 / total_records, 2)
FROM data_quality

UNION ALL

SELECT 
    'Complete Scope 2',
    total_records - missing_scope2,
    ROUND((total_records - missing_scope2) * 100.0 / total_records, 2)
FROM data_quality

UNION ALL

SELECT 
    'Valid Scope 2 Values',
    total_records - inconsistent_scope2,
    ROUND((total_records - inconsistent_scope2) * 100.0 / total_records, 2)
FROM data_quality

UNION ALL

SELECT 
    'No Negative Values',
    total_records - negative_values,
    ROUND((total_records - negative_values) * 100.0 / total_records, 2)
FROM data_quality

UNION ALL

SELECT 
    'Valid Renewable %',
    total_records - invalid_renewable,
    ROUND((total_records - invalid_renewable) * 100.0 / total_records, 2)
FROM data_quality;

-- ============================================================================
-- 8. TIME SERIES DECOMPOSITION
-- Demonstrates: Multiple window functions, trend analysis
-- ============================================================================
WITH monthly_data AS (
    SELECT 
        date,
        strftime('%Y', date) as year,
        strftime('%m', date) as month,
        SUM(scope1_tonnes + scope2_market_tonnes + scope3_tonnes) as total_emissions
    FROM emissions_monthly
    GROUP BY date
),
trend_data AS (
    SELECT 
        date,
        year,
        month,
        total_emissions,
        -- 12-month centered moving average (trend)
        AVG(total_emissions) OVER (
            ORDER BY date 
            ROWS BETWEEN 6 PRECEDING AND 6 FOLLOWING
        ) as trend,
        -- Overall average by month (seasonal component)
        AVG(total_emissions) OVER (PARTITION BY month) as seasonal_avg,
        -- Overall average (for detrending seasonal)
        AVG(total_emissions) OVER () as overall_avg
    FROM monthly_data
)
SELECT 
    date,
    ROUND(total_emissions, 0) as actual,
    ROUND(trend, 0) as trend_component,
    ROUND(seasonal_avg - overall_avg, 0) as seasonal_component,
    ROUND(total_emissions - trend, 0) as detrended,
    ROUND((total_emissions - trend) - (seasonal_avg - overall_avg), 0) as residual,
    CASE 
        WHEN trend > LAG(trend, 1) OVER (ORDER BY date) THEN 'Increasing'
        WHEN trend < LAG(trend, 1) OVER (ORDER BY date) THEN 'Decreasing'
        ELSE 'Stable'
    END as trend_direction
FROM trend_data
WHERE trend IS NOT NULL
ORDER BY date DESC
LIMIT 24;

-- ============================================================================
-- 9. RENEWABLE ENERGY IMPACT ANALYSIS
-- Demonstrates: Comparison queries, impact calculations
-- ============================================================================
SELECT 
    strftime('%Y', e.date) as year,
    f.region,
    COUNT(DISTINCT f.facility_id) as facility_count,
    ROUND(AVG(e.renewable_pct), 1) as avg_renewable_pct,
    ROUND(SUM(e.scope2_location_tonnes), 0) as scope2_location_total,
    ROUND(SUM(e.scope2_market_tonnes), 0) as scope2_market_total,
    ROUND(SUM(e.scope2_location_tonnes - e.scope2_market_tonnes), 0) as avoided_emissions,
    ROUND((SUM(e.scope2_location_tonnes - e.scope2_market_tonnes) / 
           SUM(e.scope2_location_tonnes)) * 100, 1) as avoided_emissions_pct,
    ROUND(SUM(e.electricity_mwh), 0) as total_electricity_mwh,
    ROUND(SUM(e.electricity_mwh * e.renewable_pct / 100), 0) as renewable_electricity_mwh
FROM emissions_monthly e
JOIN facilities f ON e.facility_id = f.facility_id
GROUP BY year, f.region
ORDER BY year DESC, f.region;

-- ============================================================================
-- 10. TOP OPPORTUNITIES FOR REDUCTION
-- Demonstrates: Multiple metrics, prioritization logic
-- ============================================================================
WITH facility_analysis AS (
    SELECT 
        f.facility_id,
        f.facility_name,
        f.region,
        f.facility_type,
        SUM(e.scope1_tonnes + e.scope2_market_tonnes + e.scope3_tonnes) as total_emissions,
        SUM(e.scope2_location_tonnes - e.scope2_market_tonnes) as current_renewable_impact,
        SUM(e.scope2_location_tonnes) * 0.50 as potential_renewable_impact,
        AVG(e.renewable_pct) as current_renewable_pct,
        SUM(e.scope1_tonnes) * 0.20 as potential_scope1_reduction,
        SUM(e.scope3_tonnes) * 0.30 as potential_scope3_reduction
    FROM facilities f
    JOIN emissions_monthly e ON f.facility_id = e.facility_id
    WHERE strftime('%Y', e.date) = '2023'
    GROUP BY f.facility_id
)
SELECT 
    facility_id,
    facility_name,
    region,
    facility_type,
    ROUND(total_emissions, 0) as current_emissions,
    ROUND(current_renewable_pct, 0) as current_renewable_pct,
    ROUND(potential_renewable_impact - current_renewable_impact, 0) as renewable_opportunity,
    ROUND(potential_scope1_reduction, 0) as scope1_opportunity,
    ROUND(potential_scope3_reduction, 0) as scope3_opportunity,
    ROUND(
        (potential_renewable_impact - current_renewable_impact) + 
        potential_scope1_reduction + 
        potential_scope3_reduction, 0
    ) as total_opportunity,
    ROUND(
        ((potential_renewable_impact - current_renewable_impact) + 
         potential_scope1_reduction + 
         potential_scope3_reduction) / total_emissions * 100, 1
    ) as potential_reduction_pct
FROM facility_analysis
ORDER BY total_opportunity DESC
LIMIT 10;

-- ============================================================================
-- END OF QUERY LIBRARY
-- ============================================================================

-- Usage Notes:
-- - All queries are optimized for SQLite
-- - Use appropriate indexes for production environments
-- - Adjust date filters ('2023') for current year analysis
-- - CTEs improve readability and can be materialized if needed
-- - Window functions require SQLite 3.25.0 or higher