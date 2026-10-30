# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Select a local IPv4 address before starting any teleop connections."""

import argparse
import ipaddress
import json
import os
import subprocess


def local_interfaces():
    """Return active, non-loopback IPv4 interfaces, including VPNs/bridges."""
    try:
        result = subprocess.run(
            ["ip", "-j", "-4", "addr", "show", "up"],
            capture_output=True,
            text=True,
            check=True,
        )
        records = json.loads(result.stdout)
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        raise ValueError(
            "Cannot inspect network interfaces; check that iproute2 is installed and 'ip -j -4 addr show up' works."
        ) from exc

    interfaces = {}
    for record in records:
        if "LOOPBACK" in record.get("flags", []) or record.get("operstate") == "DOWN":
            continue
        addresses = []
        for address in record.get("addr_info", []):
            if address.get("family") != "inet":
                continue
            ip = ipaddress.IPv4Address(address["local"])
            if not (ip.is_loopback or ip.is_unspecified or ip.is_multicast):
                addresses.append(str(ip))
        if addresses:
            interfaces[record["ifname"]] = list(dict.fromkeys(addresses))
    return interfaces


def select_host_ip(interface="", host_ip=""):
    interfaces = local_interfaces()
    choices = "\n".join(f"  {name}: {', '.join(ips)}" for name, ips in interfaces.items()) or "  (none)"
    hint = (
        f"Available IPv4 interfaces:\n{choices}\n"
        "Specify the interface connected to the Pico network with --interface NAME "
        "(or GENIESIM_TELEOP_INTERFACE=NAME for autoteleop.sh). "
        "Use --host-ip IP (alias --host_ip) / GENIESIM_TELEOP_HOST_IP to select a particular local IPv4 address."
    )
    if interface and host_ip:
        raise ValueError("Specify only one of --interface and --host_ip (including environment variables).")
    if host_ip:
        if not any(host_ip in ips for ips in interfaces.values()):
            raise ValueError(f"Requested IP {host_ip!r} is not on an available interface.\n{hint}")
        return host_ip
    if interface:
        if interface not in interfaces:
            raise ValueError(f"Requested interface {interface!r} is not available.\n{hint}")
        addresses = interfaces[interface]
    elif len(interfaces) == 1:
        addresses = next(iter(interfaces.values()))
    elif not interfaces:
        raise ValueError(f"No available IPv4 network interface.\n{hint}")
    else:
        raise ValueError(f"Multiple network interfaces detected; explicit selection is required.\n{hint}")
    if len(addresses) != 1:
        raise ValueError(
            f"Selected interface has multiple IPv4 addresses; specify --host-ip instead of --interface.\n{hint}"
        )
    return addresses[0]


def add_network_arguments(parser):
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--interface", default=os.environ.get("GENIESIM_TELEOP_INTERFACE", ""), help="Network interface on the Pico LAN"
    )
    group.add_argument(
        "--host_ip",
        "--host-ip",
        default=os.environ.get("GENIESIM_TELEOP_HOST_IP", ""),
        help="Local IPv4 address on the Pico LAN",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_network_arguments(parser)
    args = parser.parse_args()
    try:
        print(select_host_ip(args.interface, args.host_ip))
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
