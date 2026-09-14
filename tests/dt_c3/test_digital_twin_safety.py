from vpp.dt_c3.communication import CommunicationState
from vpp.dt_c3.digital_twin import DigitalTwinSynchronizer
from vpp.dt_c3.safety import SafetyLimits, SafetyShield


def test_digital_twin_reconstructs_missing_ess_soc():
    twin = DigitalTwinSynchronizer()
    twin.seed_state("ess", {"soc": 0.5, "capacity_kwh": 100.0}, 0)
    comm = CommunicationState("ess", step=2, delivered_step=None, delay_steps=2, aoi=2, missing=True)
    est = twin.update("ess", None, comm, {"ess_power_kw": 10.0})
    assert est.source == "predicted_missing"
    assert est.state["soc"] < 0.5


def test_safety_shield_clips_power_and_soc():
    shield = SafetyShield(SafetyLimits(ess_capacity_kwh=100.0, ess_soc_min=0.2, ess_power_max_kw=50.0))
    report = shield.project({"ess_soc": 0.21, "ev_soc": 0.5, "load_kw": 100, "pv_kw": 0, "wind_kw": 0}, {"ess_power_kw": 200.0, "ev_power_kw": 0.0, "dr_kw": 0.0})
    assert report.corrected_action["ess_power_kw"] <= 50.0
    assert report.violation_count >= 1
