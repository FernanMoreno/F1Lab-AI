-- F1Lab-AI DuckDB Schema
-- Defines canonical analytical views for simulation outputs and real data

-- Regulation configs table
CREATE TABLE IF NOT EXISTS regulations (
    regulation_id VARCHAR PRIMARY KEY,
    version VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    config JSON
);

-- Car families table
CREATE TABLE IF NOT EXISTS car_families (
    family_id VARCHAR PRIMARY KEY,
    mass_kg DOUBLE,
    cda_straight_m2 DOUBLE,
    cda_corner_m2 DOUBLE,
    cla_straight_m2 DOUBLE,
    cla_corner_m2 DOUBLE,
    power_kw DOUBLE,
    ers_efficiency DOUBLE,
    tyre_deg_factor DOUBLE,
    dirty_air_sensitivity DOUBLE,
    config JSON
);

-- Circuits metadata
CREATE TABLE IF NOT EXISTS circuits (
    circuit_id VARCHAR PRIMARY KEY,
    name VARCHAR NOT NULL,
    country VARCHAR,
    length_m DOUBLE,
    corners INT,
    drs_zones INT,
    avg_speed_kph DOUBLE,
    characteristics JSON
);

-- Simulation results
CREATE TABLE IF NOT EXISTS simulation_runs (
    run_id UUID PRIMARY KEY,
    experiment_name VARCHAR,
    regulation_id VARCHAR,
    circuit_id VARCHAR,
    seed BIGINT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    config JSON,
    metrics JSON
);

-- Lap data
CREATE TABLE IF NOT EXISTS lap_data (
    lap_id UUID PRIMARY KEY,
    run_id UUID,
    car_family_id VARCHAR,
    driver_id VARCHAR,
    lap_number INT,
    lap_time_s DOUBLE,
    sector1_s DOUBLE,
    sector2_s DOUBLE,
    sector3_s DOUBLE,
    top_speed_kph DOUBLE,
    avg_speed_kph DOUBLE,
    energy_used_mj DOUBLE,
    energy_recovered_mj DOUBLE,
    tyre_age_laps INT,
    FOREIGN KEY (run_id) REFERENCES simulation_runs(run_id)
);

-- Overtake events
CREATE TABLE IF NOT EXISTS overtakes (
    overtake_id UUID PRIMARY KEY,
    run_id UUID,
    lap_number INT,
    attacker_id VARCHAR,
    defender_id VARCHAR,
    overtake_type VARCHAR,  -- 'normal', 'drs', 'energy_boost', 'strategy'
    closing_speed_kph DOUBLE,
    energy_delta_mj DOUBLE,
    success BOOLEAN,
    location VARCHAR,
    FOREIGN KEY (run_id) REFERENCES simulation_runs(run_id)
);

-- Metric snapshots per lap
CREATE TABLE IF NOT EXISTS metric_snapshots (
    snapshot_id UUID PRIMARY KEY,
    run_id UUID,
    lap_number INT,
    battery_dependency_index DOUBLE,
    artificial_pass_index DOUBLE,
    train_formation_index DOUBLE,
    dirty_air_penalty_s DOUBLE,
    FOREIGN KEY (run_id) REFERENCES simulation_runs(run_id)
);

-- Views for dashboards

CREATE VIEW IF NOT EXISTS v_overall_health AS
SELECT
    r.regulation_id,
    AVG(m.battery_dependency_index) as avg_battery_dep,
    AVG(m.artificial_pass_index) as avg_artificial_pass,
    AVG(m.train_formation_index) as avg_train_formation,
    COUNT(DISTINCT sr.run_id) as num_runs
FROM regulations r
JOIN simulation_runs sr ON r.regulation_id = sr.regulation_id
JOIN metric_snapshots m ON sr.run_id = m.run_id
GROUP BY r.regulation_id;

CREATE VIEW IF NOT EXISTS v_circuit_stress AS
SELECT
    sr.circuit_id,
    AVG(m.train_formation_index) as avg_train_formation,
    AVG(o.closing_speed_kph) as avg_closing_speed,
    COUNT(o.overtake_id) as num_overtakes,
    AVG(CASE WHEN o.success THEN 1.0 ELSE 0.0 END) as overtake_success_rate
FROM simulation_runs sr
JOIN metric_snapshots m ON sr.run_id = m.run_id
JOIN overtakes o ON sr.run_id = o.run_id
GROUP BY sr.circuit_id;

CREATE VIEW IF NOT EXISTS v_car_family_dominance AS
SELECT
    cf.family_id,
    AVG(l.lap_time_s) as avg_lap_time,
    COUNT(DISTINCT l.run_id) as num_appearances,
    AVG(l.energy_used_mj) as avg_energy_used
FROM car_families cf
JOIN lap_data l ON cf.family_id = l.car_family_id
GROUP BY cf.family_id;