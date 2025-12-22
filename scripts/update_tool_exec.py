#!/usr/bin/env python3
"""
Update T_exec_ms values in tools_scenario*.yaml files
from cloud.json, edge.json, and device-rpi.json measurements.

Mapping:
  - cloud.json (cloud-x8aedz.xlarge) -> cloud-x8aedz.xlarge
  - edge.json (edge-nuc) -> edge-nuc
  - device-rpi.json (device-rpi) -> device-rpi
"""

import json
import yaml
from pathlib import Path


def load_measurements():
    """Load tool_exec values from measurement JSON files."""
    measurements_dir = Path(__file__).parent.parent / "measurements"

    tool_exec_data = {
        "cloud-x8aedz.xlarge": {},
        "edge-nuc": {},
        "device-rpi": {}
    }

    # Load cloud.json
    cloud_file = measurements_dir / "cloud.json"
    if cloud_file.exists():
        with open(cloud_file, 'r') as f:
            data = json.load(f)
            for tool_name, tool_data in data.get("tools", {}).items():
                tool_exec = tool_data.get("timing_ms", {}).get("tool_exec", 0)
                tool_exec_data["cloud-x8aedz.xlarge"][tool_name] = tool_exec
        print(f"Loaded {len(tool_exec_data['cloud-x8aedz.xlarge'])} tools from cloud.json")

    # Load edge.json
    edge_file = measurements_dir / "edge.json"
    if edge_file.exists():
        with open(edge_file, 'r') as f:
            data = json.load(f)
            # edge.json uses array format
            for item in data.get("measurements", []):
                tool_name = item.get("tool")
                tool_exec = item.get("tool_exec", 0)
                if tool_name:
                    tool_exec_data["edge-nuc"][tool_name] = tool_exec
        print(f"Loaded {len(tool_exec_data['edge-nuc'])} tools from edge.json")

    # Load device-rpi.json
    rpi_file = measurements_dir / "device-rpi.json"
    if rpi_file.exists():
        with open(rpi_file, 'r') as f:
            data = json.load(f)
            for tool_name, tool_data in data.get("tools", {}).items():
                tool_exec = tool_data.get("timing_ms", {}).get("tool_exec", 0)
                tool_exec_data["device-rpi"][tool_name] = tool_exec
        print(f"Loaded {len(tool_exec_data['device-rpi'])} tools from device-rpi.json")

    return tool_exec_data


def update_yaml_file(yaml_path, tool_exec_data):
    """Update T_exec_ms values in a YAML file."""
    with open(yaml_path, 'r') as f:
        content = f.read()
        data = yaml.safe_load(content)

    if not data or 'tools' not in data:
        print(f"  No tools found in {yaml_path.name}")
        return 0

    updated_count = 0

    for server_name, server_config in data['tools'].items():
        tool_profiles = server_config.get('tool_profiles', {})

        for tool_name, tool_config in tool_profiles.items():
            measurements = tool_config.get('measurements', {})
            t_exec_ms = measurements.get('T_exec_ms', {})

            if not t_exec_ms:
                continue

            updated = False
            for node_name in ['cloud-x8aedz.xlarge', 'edge-nuc', 'device-rpi']:
                if node_name in t_exec_ms and tool_name in tool_exec_data.get(node_name, {}):
                    new_value = tool_exec_data[node_name][tool_name]
                    old_value = t_exec_ms[node_name]
                    if old_value != new_value:
                        t_exec_ms[node_name] = new_value
                        updated = True
                        print(f"  {tool_name}.{node_name}: {old_value} -> {new_value}")

            if updated:
                updated_count += 1

    # Write back with preserved formatting
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    return updated_count


def main():
    print("=== Update T_exec_ms from measurement files ===\n")

    # Load measurements
    tool_exec_data = load_measurements()
    print()

    # Find all scenario YAML files
    config_dir = Path(__file__).parent.parent / "config"
    yaml_files = sorted(config_dir.glob("tools_scenario*.yaml"))

    if not yaml_files:
        print("No tools_scenario*.yaml files found")
        return

    total_updated = 0
    for yaml_path in yaml_files:
        print(f"Processing {yaml_path.name}...")
        updated = update_yaml_file(yaml_path, tool_exec_data)
        total_updated += updated
        print(f"  Updated {updated} tools\n")

    print(f"=== Done. Total tools updated: {total_updated} ===")


if __name__ == "__main__":
    main()
