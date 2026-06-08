"""Scenario generation for optimization stress tests."""

from typing import Any, Dict
import copy


def build_scenarios(base_data: Dict[str, Any], run_scenarios: bool) -> Dict[str, Dict[str, Any]]:
    """
    Create simple stress-test scenarios for the orange integration review.

    These scenarios are not replacements for the final forecast scenarios. They
    allow the optimization pipeline to be tested before the other project parts
    are completed.
    """
    scenarios = {"base": copy.deepcopy(base_data)}

    if not run_scenarios:
        return scenarios

    high_demand = copy.deepcopy(base_data)
    high_demand["demand"]["D"] = [
        float(x) * 1.25 for x in high_demand["demand"]["D"]]
    scenarios["high_demand_25pct"] = high_demand

    high_energy = copy.deepcopy(base_data)
    high_energy["costs"]["c_e"] = float(high_energy["costs"]["c_e"]) * 1.50
    scenarios["high_energy_cost_50pct"] = high_energy

    reduced_capacity = copy.deepcopy(base_data)
    reduced_capacity["server_params"]["C"] = {
        str(k): float(v) * 0.90 for k, v in reduced_capacity["server_params"]["C"].items()
    }
    scenarios["reduced_capacity_10pct"] = reduced_capacity

    return scenarios
