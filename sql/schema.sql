-- ============================================================================
-- Google Sustainability Analytics Database Schema
-- GHG Protocol Compliant Structure
-- ============================================================================

-- Drop existing tables if they exist
DROP TABLE IF EXISTS emissions_monthly;
DROP TABLE IF EXISTS business_metrics;
DROP TABLE IF EXISTS facilities;
DROP TABLE IF EXISTS emission_factors;
DROP TABLE IF EXISTS emission_targets;
DROP TABLE IF EXISTS scope3_categories;

-- ============================================================================
-- FACILITIES TABLE
-- Master data for all Google facilities (data centers and offices)
-- ============================================================================
CREATE TABLE facilities (
    facility_id TEXT PRIMARY KEY,
    facility_name TEXT NOT NULL,
    region TEXT NOT NULL,
    facility_type TEXT NOT NULL CHECK (facility_type IN ('Data Center', 'Office')),
    operational_start_year INTEGER,
    typical_renewable_pct REAL CHECK (typical_renewable_pct BETWEEN 0 AND 100),
    relative_size TEXT CHECK (relative_size IN ('Small', 'Medium', 'Large')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_facilities_region ON facilities(region);
CREATE INDEX idx_facilities_type ON facilities(facility_type);

-- ============================================================================
-- EMISSIONS_MONTHLY TABLE
-- Monthly emissions data by facility following GHG Protocol Scopes
-- ============================================================================
CREATE TABLE emissions_monthly (
    emission_id INTEGER PRIMARY KEY AUTOINCREMENT,
    facility_id TEXT NOT NULL,
    date DATE NOT NULL,
    
    -- Scope 1: Direct emissions (tonnes CO2e)
    scope1_tonnes REAL NOT NULL CHECK (scope1_tonnes >= 0),
    
    -- Scope 2: Indirect emissions from electricity (tonnes CO2e)
    scope2_location_tonnes REAL NOT NULL CHECK (scope2_location_tonnes >= 0),
    scope2_market_tonnes REAL NOT NULL CHECK (scope2_market_tonnes >= 0),
    
    -- Scope 3: Value chain emissions (tonnes CO2e)
    scope3_tonnes REAL NOT NULL CHECK (scope3_tonnes >= 0),
    
    -- Supporting metrics
    electricity_mwh REAL CHECK (electricity_mwh >= 0),
    renewable_pct REAL CHECK (renewable_pct BETWEEN 0 AND 100),
    
    -- Metadata
    data_quality_flag TEXT CHECK (data_quality_flag IN ('Verified', 'Estimated', 'Flagged')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (facility_id) REFERENCES facilities(facility_id),
    UNIQUE(facility_id, date)
);

CREATE INDEX idx_emissions_date ON emissions_monthly(date);
CREATE INDEX idx_emissions_facility ON emissions_monthly(facility_id);
CREATE INDEX idx_emissions_facility_date ON emissions_monthly(facility_id, date);

-- ============================================================================
-- BUSINESS_METRICS TABLE
-- Business activity data for emission intensity calculations
-- ============================================================================
CREATE TABLE business_metrics (
    metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
    facility_id TEXT NOT NULL,
    date DATE NOT NULL,
    
    -- Financial metrics
    revenue_millions REAL CHECK (revenue_millions >= 0),
    
    -- Activity metrics
    headcount INTEGER CHECK (headcount >= 0),
    square_feet INTEGER CHECK (square_feet >= 0),
    server_count INTEGER CHECK (server_count >= 0),
    production_volume REAL CHECK (production_volume >= 0),
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (facility_id) REFERENCES facilities(facility_id),
    UNIQUE(facility_id, date)
);

CREATE INDEX idx_business_metrics_date ON business_metrics(date);
CREATE INDEX idx_business_metrics_facility ON business_metrics(facility_id);

-- ============================================================================
-- EMISSION_FACTORS TABLE
-- Standard emission factors for GHG calculations
-- Source: EPA, EEA, DEFRA, IEA
-- ============================================================================
CREATE TABLE emission_factors (
    factor_id INTEGER PRIMARY KEY AUTOINCREMENT,
    factor_name TEXT NOT NULL,
    scope TEXT NOT NULL CHECK (scope IN ('Scope 1', 'Scope 2', 'Scope 3')),
    emission_factor REAL NOT NULL CHECK (emission_factor >= 0),
    unit TEXT NOT NULL,
    source TEXT NOT NULL,
    geography TEXT NOT NULL,
    year INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_emission_factors_scope ON emission_factors(scope);
CREATE INDEX idx_emission_factors_year ON emission_factors(year);

-- ============================================================================
-- EMISSION_TARGETS TABLE
-- Corporate emission reduction targets (Science-Based Targets)
-- Google's actual 2030 net-zero commitment
-- ============================================================================
CREATE TABLE emission_targets (
    target_id INTEGER PRIMARY KEY AUTOINCREMENT,
    scope TEXT NOT NULL,
    baseline_year INTEGER NOT NULL,
    baseline_emissions REAL NOT NULL CHECK (baseline_emissions >= 0),
    target_year INTEGER NOT NULL,
    target_description TEXT NOT NULL,
    target_emissions REAL NOT NULL CHECK (target_emissions >= 0),
    sbti_aligned BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    CHECK (target_year > baseline_year)
);

-- ============================================================================
-- SCOPE3_CATEGORIES TABLE
-- GHG Protocol Scope 3 standard categories
-- 15 categories covering full value chain
-- ============================================================================
CREATE TABLE scope3_categories (
    category_id INTEGER PRIMARY KEY CHECK (category_id BETWEEN 1 AND 15),
    category_name TEXT NOT NULL,
    description TEXT NOT NULL,
    typical_pct_of_scope3 REAL CHECK (typical_pct_of_scope3 >= 0),
    upstream_downstream TEXT CHECK (upstream_downstream IN ('Upstream', 'Downstream')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- VIEWS FOR COMMON QUERIES
-- ============================================================================

-- Total emissions by facility and month
CREATE VIEW vw_monthly_totals AS
SELECT 
    e.facility_id,
    f.facility_name,
    f.region,
    f.facility_type,
    e.date,
    strftime('%Y', e.date) as year,
    strftime('%Y-%m', e.date) as month,
    e.scope1_tonnes,
    e.scope2_location_tonnes,
    e.scope2_market_tonnes,
    e.scope3_tonnes,
    (e.scope1_tonnes + e.scope2_market_tonnes + e.scope3_tonnes) as total_emissions,
    e.renewable_pct,
    b.revenue_millions,
    CASE 
        WHEN b.revenue_millions > 0 
        THEN (e.scope1_tonnes + e.scope2_market_tonnes + e.scope3_tonnes) / b.revenue_millions 
        ELSE NULL 
    END as emission_intensity
FROM emissions_monthly e
JOIN facilities f ON e.facility_id = f.facility_id
LEFT JOIN business_metrics b ON e.facility_id = b.facility_id AND e.date = b.date;

-- Annual emissions summary
CREATE VIEW vw_annual_summary AS
SELECT 
    strftime('%Y', date) as year,
    COUNT(DISTINCT facility_id) as facility_count,
    SUM(scope1_tonnes) as total_scope1,
    SUM(scope2_location_tonnes) as total_scope2_location,
    SUM(scope2_market_tonnes) as total_scope2_market,
    SUM(scope3_tonnes) as total_scope3,
    SUM(scope1_tonnes + scope2_market_tonnes + scope3_tonnes) as total_emissions,
    AVG(renewable_pct) as avg_renewable_pct
FROM emissions_monthly
GROUP BY year
ORDER BY year;

-- Facility rankings by emission intensity
CREATE VIEW vw_facility_rankings AS
SELECT 
    f.facility_id,
    f.facility_name,
    f.region,
    f.facility_type,
    SUM(e.scope1_tonnes + e.scope2_market_tonnes + e.scope3_tonnes) as total_emissions,
    SUM(b.revenue_millions) as total_revenue,
    CASE 
        WHEN SUM(b.revenue_millions) > 0 
        THEN SUM(e.scope1_tonnes + e.scope2_market_tonnes + e.scope3_tonnes) / SUM(b.revenue_millions)
        ELSE NULL 
    END as emission_intensity,
    AVG(e.renewable_pct) as avg_renewable_pct,
    RANK() OVER (ORDER BY 
        CASE 
            WHEN SUM(b.revenue_millions) > 0 
            THEN SUM(e.scope1_tonnes + e.scope2_market_tonnes + e.scope3_tonnes) / SUM(b.revenue_millions)
            ELSE 999999 
        END ASC
    ) as intensity_rank
FROM facilities f
JOIN emissions_monthly e ON f.facility_id = e.facility_id
JOIN business_metrics b ON f.facility_id = b.facility_id AND e.date = b.date
WHERE strftime('%Y', e.date) = strftime('%Y', 'now')
GROUP BY f.facility_id
ORDER BY emission_intensity ASC;

-- ============================================================================
-- TRIGGERS FOR UPDATED_AT TIMESTAMPS
-- ============================================================================

CREATE TRIGGER update_facilities_timestamp 
AFTER UPDATE ON facilities
BEGIN
    UPDATE facilities SET updated_at = CURRENT_TIMESTAMP WHERE facility_id = NEW.facility_id;
END;

CREATE TRIGGER update_emissions_timestamp 
AFTER UPDATE ON emissions_monthly
BEGIN
    UPDATE emissions_monthly SET updated_at = CURRENT_TIMESTAMP WHERE emission_id = NEW.emission_id;
END;

CREATE TRIGGER update_business_metrics_timestamp 
AFTER UPDATE ON business_metrics
BEGIN
    UPDATE business_metrics SET updated_at = CURRENT_TIMESTAMP WHERE metric_id = NEW.metric_id;
END;

CREATE TRIGGER update_targets_timestamp 
AFTER UPDATE ON emission_targets
BEGIN
    UPDATE emission_targets SET updated_at = CURRENT_TIMESTAMP WHERE target_id = NEW.target_id;
END;

-- ============================================================================
-- DATA VALIDATION TRIGGERS
-- ============================================================================

-- Ensure Scope 2 market-based <= location-based
CREATE TRIGGER validate_scope2_emissions
BEFORE INSERT ON emissions_monthly
BEGIN
    SELECT CASE
        WHEN NEW.scope2_market_tonnes > NEW.scope2_location_tonnes THEN
            RAISE(ABORT, 'Scope 2 market-based emissions cannot exceed location-based emissions')
    END;
END;

-- Ensure date is not in the future
CREATE TRIGGER validate_emission_date
BEFORE INSERT ON emissions_monthly
BEGIN
    SELECT CASE
        WHEN NEW.date > date('now') THEN
            RAISE(ABORT, 'Emission date cannot be in the future')
    END;
END;

-- ============================================================================
-- COMMENTS (SQLite supports them via schema metadata)
-- ============================================================================

-- This schema implements:
-- 1. GHG Protocol Scopes 1, 2 (location & market-based), and 3
-- 2. Proper normalization (facilities as master data)
-- 3. Data quality constraints (CHECK clauses)
-- 4. Performance indexes on common query patterns
-- 5. Audit trails (created_at, updated_at)
-- 6. Referential integrity (FOREIGN KEY constraints)
-- 7. Views for common analytical queries
-- 8. Triggers for data validation and timestamp management

-- ============================================================================
-- END OF SCHEMA
-- ============================================================================