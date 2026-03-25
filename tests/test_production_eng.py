"""Tests for production engineering tools."""

import json
import math

import pytest

from petro_mcp.tools.production_eng import (
    calculate_beggs_brill_pressure_drop,
    calculate_coleman_critical_rate,
    calculate_critical_choke_flow,
    calculate_erosional_velocity,
    calculate_hydrate_inhibitor_dosing,
    calculate_hydrate_temperature,
    calculate_turner_critical_rate,
)


# ---------------------------------------------------------------------------
# Beggs & Brill
# ---------------------------------------------------------------------------

class TestBeggsBrill:
    """Tests for Beggs & Brill multiphase pressure drop."""

    def test_vertical_oil_well(self):
        """Typical vertical oil well — should produce reasonable BHP."""
        result = json.loads(calculate_beggs_brill_pressure_drop(
            flow_rate_bpd=1000,
            gor_scf_bbl=500,
            water_cut=0.3,
            oil_api=35,
            gas_sg=0.65,
            pipe_id_in=2.441,
            pipe_length_ft=8000,
            inclination_deg=90,
            wellhead_pressure_psi=200,
            temperature_f=180,
        ))
        assert result["flowing_bhp_psi"] > result["inputs"]["wellhead_pressure_psi"]
        assert result["pressure_drop_psi"] > 0
        assert result["flow_pattern"] in ("segregated", "intermittent", "distributed", "transition")
        assert 0 <= result["liquid_holdup"] <= 1
        assert result["correlation"] == "Beggs and Brill (1973)"

    def test_horizontal_pipe(self):
        """Horizontal pipe — no elevation component."""
        result = json.loads(calculate_beggs_brill_pressure_drop(
            flow_rate_bpd=2000,
            gor_scf_bbl=300,
            water_cut=0.5,
            oil_api=30,
            gas_sg=0.7,
            pipe_id_in=4.0,
            pipe_length_ft=5000,
            inclination_deg=0,
            wellhead_pressure_psi=500,
            temperature_f=150,
        ))
        # Horizontal: elevation gradient should be ~0
        assert abs(result["elevation_gradient_psi_ft"]) < 0.001
        assert result["pressure_drop_psi"] > 0
        assert result["friction_gradient_psi_ft"] > 0

    def test_high_gor_gas_well(self):
        """High GOR well — should be distributed or intermittent."""
        result = json.loads(calculate_beggs_brill_pressure_drop(
            flow_rate_bpd=200,
            gor_scf_bbl=5000,
            water_cut=0.1,
            oil_api=45,
            gas_sg=0.6,
            pipe_id_in=2.875,
            pipe_length_ft=10000,
            inclination_deg=90,
            wellhead_pressure_psi=500,
            temperature_f=200,
        ))
        assert result["flowing_bhp_psi"] > 500
        assert result["superficial_gas_velocity_ft_s"] > result["superficial_liquid_velocity_ft_s"]

    def test_downward_inclination(self):
        """Downward flow — pressure drop should be less than vertical up."""
        result_up = json.loads(calculate_beggs_brill_pressure_drop(
            flow_rate_bpd=500, gor_scf_bbl=400, water_cut=0.2,
            oil_api=35, gas_sg=0.65, pipe_id_in=2.441,
            pipe_length_ft=5000, inclination_deg=90,
            wellhead_pressure_psi=300, temperature_f=180,
        ))
        result_down = json.loads(calculate_beggs_brill_pressure_drop(
            flow_rate_bpd=500, gor_scf_bbl=400, water_cut=0.2,
            oil_api=35, gas_sg=0.65, pipe_id_in=2.441,
            pipe_length_ft=5000, inclination_deg=-45,
            wellhead_pressure_psi=300, temperature_f=180,
        ))
        assert result_up["pressure_drop_psi"] > result_down["pressure_drop_psi"]

    def test_returns_all_expected_keys(self):
        result = json.loads(calculate_beggs_brill_pressure_drop(
            flow_rate_bpd=1000, gor_scf_bbl=500, water_cut=0.3,
            oil_api=35, gas_sg=0.65, pipe_id_in=2.441,
            pipe_length_ft=8000, inclination_deg=90,
            wellhead_pressure_psi=200, temperature_f=180,
        ))
        expected_keys = {
            "inputs", "correlation", "units", "flow_pattern",
            "liquid_holdup", "no_slip_holdup", "mixture_velocity_ft_s",
            "superficial_liquid_velocity_ft_s", "superficial_gas_velocity_ft_s",
            "two_phase_friction_factor", "reynolds_number",
            "pressure_gradient_psi_ft", "elevation_gradient_psi_ft",
            "friction_gradient_psi_ft", "pressure_drop_psi", "flowing_bhp_psi",
            "slip_density_lb_ft3", "no_slip_density_lb_ft3",
        }
        assert expected_keys.issubset(set(result.keys()))

    def test_invalid_water_cut(self):
        with pytest.raises(ValueError, match="water_cut"):
            calculate_beggs_brill_pressure_drop(
                flow_rate_bpd=1000, gor_scf_bbl=500, water_cut=1.5,
                oil_api=35, gas_sg=0.65, pipe_id_in=2.441,
                pipe_length_ft=8000, inclination_deg=90,
                wellhead_pressure_psi=200, temperature_f=180,
            )

    def test_invalid_inclination(self):
        with pytest.raises(ValueError, match="inclination"):
            calculate_beggs_brill_pressure_drop(
                flow_rate_bpd=1000, gor_scf_bbl=500, water_cut=0.3,
                oil_api=35, gas_sg=0.65, pipe_id_in=2.441,
                pipe_length_ft=8000, inclination_deg=100,
                wellhead_pressure_psi=200, temperature_f=180,
            )

    def test_zero_flow_rate_raises(self):
        with pytest.raises(ValueError, match="positive"):
            calculate_beggs_brill_pressure_drop(
                flow_rate_bpd=0, gor_scf_bbl=500, water_cut=0.3,
                oil_api=35, gas_sg=0.65, pipe_id_in=2.441,
                pipe_length_ft=8000, inclination_deg=90,
                wellhead_pressure_psi=200, temperature_f=180,
            )

    def test_pure_oil_no_gas(self):
        """Zero GOR — all liquid flow."""
        result = json.loads(calculate_beggs_brill_pressure_drop(
            flow_rate_bpd=500, gor_scf_bbl=0, water_cut=0.0,
            oil_api=30, gas_sg=0.65, pipe_id_in=2.441,
            pipe_length_ft=8000, inclination_deg=90,
            wellhead_pressure_psi=200, temperature_f=150,
        ))
        assert result["flowing_bhp_psi"] > 200
        assert result["no_slip_holdup"] == 1.0

    def test_pure_water(self):
        """100% water cut."""
        result = json.loads(calculate_beggs_brill_pressure_drop(
            flow_rate_bpd=1000, gor_scf_bbl=500, water_cut=1.0,
            oil_api=35, gas_sg=0.65, pipe_id_in=2.441,
            pipe_length_ft=8000, inclination_deg=90,
            wellhead_pressure_psi=200, temperature_f=180,
        ))
        # With WC=1.0, oil_rate=0 so gor doesn't produce free gas
        assert result["flowing_bhp_psi"] > 200


# ---------------------------------------------------------------------------
# Turner Critical Rate
# ---------------------------------------------------------------------------

class TestTurnerCritical:
    """Tests for Turner et al. (1969) liquid loading critical rate."""

    def test_basic_calculation(self):
        result = json.loads(calculate_turner_critical_rate(
            wellhead_pressure_psi=300,
            wellhead_temp_f=120,
            gas_sg=0.65,
        ))
        assert result["critical_velocity_water_ft_s"] > 0
        assert result["critical_rate_water_mcfd"] > 0
        assert result["correlation"] == "Turner et al. (1969)"
        assert result["gas_density_lb_ft3"] > 0

    def test_higher_pressure_higher_rate(self):
        """Higher wellhead pressure requires higher critical rate."""
        r_low = json.loads(calculate_turner_critical_rate(
            wellhead_pressure_psi=200, wellhead_temp_f=120, gas_sg=0.65,
        ))
        r_high = json.loads(calculate_turner_critical_rate(
            wellhead_pressure_psi=800, wellhead_temp_f=120, gas_sg=0.65,
        ))
        assert r_high["critical_rate_water_mcfd"] > r_low["critical_rate_water_mcfd"]

    def test_with_condensate(self):
        result = json.loads(calculate_turner_critical_rate(
            wellhead_pressure_psi=500,
            wellhead_temp_f=150,
            gas_sg=0.7,
            condensate_sg=0.78,
        ))
        assert "critical_velocity_condensate_ft_s" in result
        assert "critical_rate_condensate_mcfd" in result
        # Condensate has lower surface tension, so critical velocity should be lower
        assert result["critical_velocity_condensate_ft_s"] < result["critical_velocity_water_ft_s"]

    def test_status_unloading(self):
        result = json.loads(calculate_turner_critical_rate(
            wellhead_pressure_psi=300, wellhead_temp_f=120, gas_sg=0.65,
            current_rate_mcfd=5000,
        ))
        assert "unloading" in result["status"]

    def test_status_loading(self):
        result = json.loads(calculate_turner_critical_rate(
            wellhead_pressure_psi=300, wellhead_temp_f=120, gas_sg=0.65,
            current_rate_mcfd=10,
        ))
        assert "loading" in result["status"]

    def test_larger_tubing_lower_rate(self):
        """Larger tubing means lower velocity for same rate, so higher critical rate needed."""
        r_small = json.loads(calculate_turner_critical_rate(
            wellhead_pressure_psi=300, wellhead_temp_f=120, gas_sg=0.65,
            tubing_id_in=2.441,
        ))
        r_large = json.loads(calculate_turner_critical_rate(
            wellhead_pressure_psi=300, wellhead_temp_f=120, gas_sg=0.65,
            tubing_id_in=3.5,
        ))
        assert r_large["critical_rate_water_mcfd"] > r_small["critical_rate_water_mcfd"]

    def test_invalid_pressure(self):
        with pytest.raises(ValueError, match="positive"):
            calculate_turner_critical_rate(
                wellhead_pressure_psi=0, wellhead_temp_f=120, gas_sg=0.65,
            )


# ---------------------------------------------------------------------------
# Coleman Critical Rate
# ---------------------------------------------------------------------------

class TestColemanCritical:
    """Tests for Coleman et al. (1991) critical rate."""

    def test_basic_calculation(self):
        result = json.loads(calculate_coleman_critical_rate(
            wellhead_pressure_psi=300,
            wellhead_temp_f=120,
            gas_sg=0.65,
        ))
        assert result["critical_velocity_ft_s"] > 0
        assert result["critical_rate_mcfd"] > 0
        assert result["correlation"] == "Coleman et al. (1991)"

    def test_twenty_percent_below_turner(self):
        """Coleman should be exactly 80% of Turner velocity."""
        turner = json.loads(calculate_turner_critical_rate(
            wellhead_pressure_psi=300, wellhead_temp_f=120, gas_sg=0.65,
        ))
        coleman = json.loads(calculate_coleman_critical_rate(
            wellhead_pressure_psi=300, wellhead_temp_f=120, gas_sg=0.65,
        ))
        assert abs(coleman["critical_velocity_ft_s"] - 0.8 * turner["critical_velocity_water_ft_s"]) < 0.01
        assert coleman["critical_rate_mcfd"] < turner["critical_rate_water_mcfd"]

    def test_status_check(self):
        result = json.loads(calculate_coleman_critical_rate(
            wellhead_pressure_psi=300, wellhead_temp_f=120, gas_sg=0.65,
            current_rate_mcfd=5000,
        ))
        assert "status" in result


# ---------------------------------------------------------------------------
# Hydrate Temperature
# ---------------------------------------------------------------------------

class TestHydrateTemperature:
    """Tests for hydrate temperature prediction."""

    def test_basic_calculation(self):
        result = json.loads(calculate_hydrate_temperature(
            pressure_psi=1000, gas_sg=0.65,
        ))
        assert "hydrate_temperature_f" in result
        assert result["correlation"] == "Katz gas-gravity chart (empirical fit)"

    def test_higher_pressure_higher_temp(self):
        """Higher pressure = higher hydrate formation temperature."""
        r_low = json.loads(calculate_hydrate_temperature(pressure_psi=500, gas_sg=0.65))
        r_high = json.loads(calculate_hydrate_temperature(pressure_psi=2000, gas_sg=0.65))
        assert r_high["hydrate_temperature_f"] > r_low["hydrate_temperature_f"]

    def test_heavier_gas_higher_temp(self):
        """Heavier gas = higher hydrate temperature."""
        r_light = json.loads(calculate_hydrate_temperature(pressure_psi=1000, gas_sg=0.6))
        r_heavy = json.loads(calculate_hydrate_temperature(pressure_psi=1000, gas_sg=0.9))
        assert r_heavy["hydrate_temperature_f"] > r_light["hydrate_temperature_f"]

    def test_invalid_gas_sg(self):
        with pytest.raises(ValueError, match="gas_sg"):
            calculate_hydrate_temperature(pressure_psi=1000, gas_sg=0.3)

    def test_invalid_pressure(self):
        with pytest.raises(ValueError, match="positive"):
            calculate_hydrate_temperature(pressure_psi=0, gas_sg=0.65)

    def test_reasonable_range(self):
        """At typical pipeline conditions, hydrate temp should be 30-80 F."""
        result = json.loads(calculate_hydrate_temperature(pressure_psi=1000, gas_sg=0.65))
        t = result["hydrate_temperature_f"]
        assert 30 < t < 80


# ---------------------------------------------------------------------------
# Hydrate Inhibitor Dosing
# ---------------------------------------------------------------------------

class TestHydrateInhibitorDosing:
    """Tests for Hammerschmidt inhibitor dosing."""

    def test_methanol_dosing(self):
        result = json.loads(calculate_hydrate_inhibitor_dosing(
            hydrate_temp_f=65, operating_temp_f=40, water_rate_bwpd=100,
        ))
        assert result["inhibitor_weight_pct"] > 0
        assert result["inhibitor_rate_gal_day"] > 0
        assert result["temperature_depression_f"] == 25.0
        assert result["correlation"] == "Hammerschmidt (1934)"

    def test_meg_dosing(self):
        result = json.loads(calculate_hydrate_inhibitor_dosing(
            hydrate_temp_f=65, operating_temp_f=40, water_rate_bwpd=100,
            inhibitor="meg",
        ))
        assert result["inhibitor_weight_pct"] > 0

    def test_ethanol_dosing(self):
        result = json.loads(calculate_hydrate_inhibitor_dosing(
            hydrate_temp_f=65, operating_temp_f=40, water_rate_bwpd=100,
            inhibitor="ethanol",
        ))
        assert result["inhibitor_weight_pct"] > 0

    def test_no_inhibitor_needed(self):
        """Operating temp below hydrate temp — no inhibitor needed."""
        result = json.loads(calculate_hydrate_inhibitor_dosing(
            hydrate_temp_f=40, operating_temp_f=60, water_rate_bwpd=100,
        ))
        assert result["inhibitor_weight_pct"] == 0.0
        assert result["inhibitor_rate_gal_day"] == 0.0

    def test_larger_dt_more_inhibitor(self):
        """Bigger temperature depression = more inhibitor."""
        r_small = json.loads(calculate_hydrate_inhibitor_dosing(
            hydrate_temp_f=65, operating_temp_f=55, water_rate_bwpd=100,
        ))
        r_large = json.loads(calculate_hydrate_inhibitor_dosing(
            hydrate_temp_f=65, operating_temp_f=30, water_rate_bwpd=100,
        ))
        assert r_large["inhibitor_weight_pct"] > r_small["inhibitor_weight_pct"]
        assert r_large["inhibitor_rate_gal_day"] > r_small["inhibitor_rate_gal_day"]

    def test_invalid_inhibitor(self):
        with pytest.raises(ValueError, match="Unsupported"):
            calculate_hydrate_inhibitor_dosing(
                hydrate_temp_f=65, operating_temp_f=40, water_rate_bwpd=100,
                inhibitor="glycerol",
            )

    def test_invalid_water_rate(self):
        with pytest.raises(ValueError, match="positive"):
            calculate_hydrate_inhibitor_dosing(
                hydrate_temp_f=65, operating_temp_f=40, water_rate_bwpd=0,
            )


# ---------------------------------------------------------------------------
# Erosional Velocity
# ---------------------------------------------------------------------------

class TestErosionalVelocity:
    """Tests for API RP 14E erosional velocity."""

    def test_basic_calculation(self):
        result = json.loads(calculate_erosional_velocity(
            density_mix_lb_ft3=10.0, c_factor=100,
        ))
        expected = 100.0 / math.sqrt(10.0)
        assert abs(result["erosional_velocity_ft_s"] - expected) < 0.01
        assert result["correlation"] == "API RP 14E"

    def test_higher_density_lower_velocity(self):
        r_light = json.loads(calculate_erosional_velocity(density_mix_lb_ft3=5.0))
        r_heavy = json.loads(calculate_erosional_velocity(density_mix_lb_ft3=30.0))
        assert r_light["erosional_velocity_ft_s"] > r_heavy["erosional_velocity_ft_s"]

    def test_higher_c_factor_higher_velocity(self):
        r_100 = json.loads(calculate_erosional_velocity(density_mix_lb_ft3=10.0, c_factor=100))
        r_150 = json.loads(calculate_erosional_velocity(density_mix_lb_ft3=10.0, c_factor=150))
        assert r_150["erosional_velocity_ft_s"] > r_100["erosional_velocity_ft_s"]

    def test_invalid_density(self):
        with pytest.raises(ValueError, match="positive"):
            calculate_erosional_velocity(density_mix_lb_ft3=0)

    def test_known_value(self):
        """C=100, rho=4 lb/ft3 -> v_e = 50 ft/s."""
        result = json.loads(calculate_erosional_velocity(
            density_mix_lb_ft3=4.0, c_factor=100,
        ))
        assert abs(result["erosional_velocity_ft_s"] - 50.0) < 0.01


# ---------------------------------------------------------------------------
# Gilbert Choke Flow
# ---------------------------------------------------------------------------

class TestCriticalChokeFlow:
    """Tests for Gilbert choke correlation."""

    def test_basic_calculation(self):
        result = json.loads(calculate_critical_choke_flow(
            upstream_pressure_psi=1000,
            choke_size_64ths=32,
            gor_scf_bbl=500,
            oil_api=35,
        ))
        assert result["total_liquid_rate_bpd"] > 0
        assert result["oil_rate_bopd"] > 0
        assert result["gas_rate_mcfd"] > 0
        assert result["correlation"] == "Gilbert (1954)"

    def test_larger_choke_higher_rate(self):
        r_small = json.loads(calculate_critical_choke_flow(
            upstream_pressure_psi=1000, choke_size_64ths=16,
            gor_scf_bbl=500, oil_api=35,
        ))
        r_large = json.loads(calculate_critical_choke_flow(
            upstream_pressure_psi=1000, choke_size_64ths=48,
            gor_scf_bbl=500, oil_api=35,
        ))
        assert r_large["total_liquid_rate_bpd"] > r_small["total_liquid_rate_bpd"]

    def test_higher_pressure_higher_rate(self):
        r_low = json.loads(calculate_critical_choke_flow(
            upstream_pressure_psi=500, choke_size_64ths=32,
            gor_scf_bbl=500, oil_api=35,
        ))
        r_high = json.loads(calculate_critical_choke_flow(
            upstream_pressure_psi=2000, choke_size_64ths=32,
            gor_scf_bbl=500, oil_api=35,
        ))
        assert r_high["total_liquid_rate_bpd"] > r_low["total_liquid_rate_bpd"]

    def test_with_water_cut(self):
        result = json.loads(calculate_critical_choke_flow(
            upstream_pressure_psi=1000, choke_size_64ths=32,
            gor_scf_bbl=500, oil_api=35, water_cut=0.3,
        ))
        assert result["water_rate_bwpd"] > 0
        assert result["oil_rate_bopd"] < result["total_liquid_rate_bpd"]

    def test_choke_size_inches(self):
        """32/64 = 0.5 inches."""
        result = json.loads(calculate_critical_choke_flow(
            upstream_pressure_psi=1000, choke_size_64ths=32,
            gor_scf_bbl=500, oil_api=35,
        ))
        assert abs(result["choke_size_inches"] - 0.5) < 0.001

    def test_invalid_pressure(self):
        with pytest.raises(ValueError, match="positive"):
            calculate_critical_choke_flow(
                upstream_pressure_psi=0, choke_size_64ths=32,
                gor_scf_bbl=500, oil_api=35,
            )

    def test_invalid_water_cut(self):
        with pytest.raises(ValueError, match="water_cut"):
            calculate_critical_choke_flow(
                upstream_pressure_psi=1000, choke_size_64ths=32,
                gor_scf_bbl=500, oil_api=35, water_cut=1.5,
            )

    def test_gilbert_formula_directly(self):
        """Verify against hand calculation: q = P * S^1.89 / (435 * GLR^0.546)."""
        p, s, glr = 1000.0, 32.0, 500.0
        expected = p * s ** 1.89 / (435.0 * glr ** 0.546)
        result = json.loads(calculate_critical_choke_flow(
            upstream_pressure_psi=p, choke_size_64ths=s,
            gor_scf_bbl=glr, oil_api=35, water_cut=0.0,
        ))
        assert abs(result["total_liquid_rate_bpd"] - expected) < 1.0
