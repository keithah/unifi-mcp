"""Tests for unifi_update_traffic_route editing route-match fields (target_devices, domains, etc.)."""

import copy
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

os.environ.setdefault("UNIFI_HOST", "127.0.0.1")
os.environ.setdefault("UNIFI_USERNAME", "test")
os.environ.setdefault("UNIFI_PASSWORD", "test")


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

SAMPLE_ROUTE = {
    "_id": "route-001",
    "description": "secondlife-staging.com",
    "enabled": True,
    "kill_switch_enabled": False,
    "network_id": "net-vpn",
    "matching_target": "DOMAIN",
    "domains": [{"domain": "secondlife-staging.com", "ports": [], "port_ranges": []}],
    "ip_addresses": [],
    "ip_ranges": [],
    "regions": [],
    "target_devices": [{"type": "CLIENT", "client_mac": "dc:cc:e6:66:86:2b"}],
    "next_hop": "",
}

NEW_TARGETS = [{"type": "CLIENT", "client_mac": "fe:38:bd:88:e9:c5"}]


def _mock_manager():
    """Return a mock TrafficRouteManager with async get/update methods."""
    mgr = MagicMock()
    mgr.get_traffic_route_details = AsyncMock(return_value=copy.deepcopy(SAMPLE_ROUTE))
    mgr.update_traffic_route = AsyncMock(return_value=True)
    mgr.create_traffic_route = AsyncMock(return_value={"_id": "route-new", "description": "YouTube via VPN"})
    return mgr


class TestCreateTrafficRoute:
    @pytest.mark.asyncio
    async def test_domain_route_preview_does_not_mutate(self):
        mgr = _mock_manager()
        with patch("unifi_network_mcp.tools.traffic_routes.traffic_route_manager", mgr):
            from unifi_network_mcp.tools.traffic_routes import create_traffic_route

            result = await create_traffic_route(
                description="YouTube via VPN",
                matching_target="DOMAIN",
                network_id="vpn-albania",
                domains=[{"domain": "youtube.com", "ports": [], "port_ranges": []}],
            )

        assert result["success"] is True
        assert result["requires_confirmation"] is True
        assert result["preview"]["will_create"]["matching_target"] == "DOMAIN"
        mgr.create_traffic_route.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_domain_route_confirm_posts_normalized_payload(self):
        mgr = _mock_manager()
        with patch("unifi_network_mcp.tools.traffic_routes.traffic_route_manager", mgr):
            from unifi_network_mcp.tools.traffic_routes import create_traffic_route

            result = await create_traffic_route(
                description="YouTube via VPN",
                matching_target="DOMAIN",
                network_id="vpn-albania",
                domains=[{"domain": "youtube.com"}],
                target_devices=[{"type": "ALL_CLIENTS"}],
                kill_switch_enabled=True,
                confirm=True,
            )

        assert result["success"] is True
        assert result["route_id"] == "route-new"
        mgr.create_traffic_route.assert_awaited_once_with(
            {
                "description": "YouTube via VPN",
                "matching_target": "DOMAIN",
                "network_id": "vpn-albania",
                "domains": [{"domain": "youtube.com", "ports": [], "port_ranges": []}],
                "target_devices": [{"type": "ALL_CLIENTS"}],
                "kill_switch_enabled": True,
                "enabled": True,
                "ip_addresses": [],
                "ip_ranges": [],
                "regions": [],
                "next_hop": "",
            }
        )

    @pytest.mark.asyncio
    async def test_domain_route_rejects_missing_domains(self):
        mgr = _mock_manager()
        with patch("unifi_network_mcp.tools.traffic_routes.traffic_route_manager", mgr):
            from unifi_network_mcp.tools.traffic_routes import create_traffic_route

            result = await create_traffic_route(
                description="Unsafe route",
                matching_target="DOMAIN",
                network_id="vpn-albania",
                confirm=True,
            )

        assert result["success"] is False
        assert "domains" in result["error"]
        mgr.create_traffic_route.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_internet_route_with_single_client_target_previews(self):
        mgr = _mock_manager()
        networks = MagicMock()
        networks.get_network_details = AsyncMock(return_value={"_id": "wan-att", "purpose": "wan"})
        target = [{"type": "CLIENT", "client_mac": "aa:bb:cc:dd:ee:ff"}]
        with (
            patch("unifi_network_mcp.tools.traffic_routes.traffic_route_manager", mgr),
            patch("unifi_network_mcp.tools.traffic_routes.network_manager", networks),
        ):
            from unifi_network_mcp.tools.traffic_routes import create_traffic_route

            result = await create_traffic_route(
                description="Desktop via WAN2",
                matching_target="INTERNET",
                network_id="wan-att",
                target_devices=target,
                kill_switch_enabled=True,
            )

        assert result["success"] is True
        assert result["requires_confirmation"] is True
        assert result["preview"]["will_create"]["matching_target"] == "INTERNET"
        assert result["preview"]["will_create"]["target_devices"] == target
        mgr.create_traffic_route.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_internet_route_rejects_non_wan_target(self):
        mgr = _mock_manager()
        networks = MagicMock()
        networks.get_network_details = AsyncMock(return_value={"_id": "vpn-route", "purpose": "remote-user-vpn"})
        with (
            patch("unifi_network_mcp.tools.traffic_routes.traffic_route_manager", mgr),
            patch("unifi_network_mcp.tools.traffic_routes.network_manager", networks, create=True),
        ):
            from unifi_network_mcp.tools.traffic_routes import create_traffic_route

            result = await create_traffic_route(
                description="Unsafe route",
                matching_target="INTERNET",
                network_id="vpn-route",
                target_devices=[{"type": "CLIENT", "client_mac": "aa:bb:cc:dd:ee:ff"}],
            )

        assert result["success"] is False
        assert "WAN network" in result["error"]
        mgr.create_traffic_route.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_internet_route_rejects_all_clients_target(self):
        mgr = _mock_manager()
        networks = MagicMock()
        networks.get_network_details = AsyncMock(return_value={"_id": "wan-att", "purpose": "wan"})
        with (
            patch("unifi_network_mcp.tools.traffic_routes.traffic_route_manager", mgr),
            patch("unifi_network_mcp.tools.traffic_routes.network_manager", networks),
        ):
            from unifi_network_mcp.tools.traffic_routes import create_traffic_route

            result = await create_traffic_route(
                description="Unsafe route",
                matching_target="INTERNET",
                network_id="wan-att",
                target_devices=[{"type": "ALL_CLIENTS"}],
            )

        assert result["success"] is False
        assert "single CLIENT" in result["error"]
        mgr.create_traffic_route.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_internet_route_rejects_invalid_client_mac(self):
        mgr = _mock_manager()
        networks = MagicMock()
        networks.get_network_details = AsyncMock(return_value={"_id": "wan-att", "purpose": "wan"})
        with (
            patch("unifi_network_mcp.tools.traffic_routes.traffic_route_manager", mgr),
            patch("unifi_network_mcp.tools.traffic_routes.network_manager", networks),
        ):
            from unifi_network_mcp.tools.traffic_routes import create_traffic_route

            result = await create_traffic_route(
                description="Unsafe route",
                matching_target="INTERNET",
                network_id="wan-att",
                target_devices=[{"type": "CLIENT", "client_mac": "not-a-mac"}],
            )

        assert result["success"] is False
        assert "valid client_mac" in result["error"]
        mgr.create_traffic_route.assert_not_awaited()


# ---------------------------------------------------------------------------
# Preview / apply for target_devices (the headline use case: swapping a device)
# ---------------------------------------------------------------------------


class TestUpdateTrafficRouteTargets:
    @pytest.mark.asyncio
    async def test_target_devices_preview_shows_current_and_proposed(self):
        mgr = _mock_manager()
        with patch("unifi_network_mcp.tools.traffic_routes.traffic_route_manager", mgr):
            from unifi_network_mcp.tools.traffic_routes import update_traffic_route

            result = await update_traffic_route("route-001", target_devices=NEW_TARGETS, confirm=False)

        assert result["success"] is True
        assert result["requires_confirmation"] is True
        assert result["preview"]["current"]["target_devices"] == SAMPLE_ROUTE["target_devices"]
        assert result["preview"]["proposed"]["target_devices"] == NEW_TARGETS
        mgr.update_traffic_route.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_target_devices_apply_forwards_to_manager(self):
        mgr = _mock_manager()
        with patch("unifi_network_mcp.tools.traffic_routes.traffic_route_manager", mgr):
            from unifi_network_mcp.tools.traffic_routes import update_traffic_route

            result = await update_traffic_route("route-001", target_devices=NEW_TARGETS, confirm=True)

        assert result["success"] is True
        mgr.update_traffic_route.assert_awaited_once_with("route-001", target_devices=NEW_TARGETS)

    @pytest.mark.asyncio
    async def test_multiple_route_match_fields_forwarded(self):
        mgr = _mock_manager()
        new_domains = [{"domain": "dev.secondlife.io", "ports": [], "port_ranges": []}]
        with patch("unifi_network_mcp.tools.traffic_routes.traffic_route_manager", mgr):
            from unifi_network_mcp.tools.traffic_routes import update_traffic_route

            result = await update_traffic_route(
                "route-001",
                domains=new_domains,
                regions=["US"],
                next_hop="10.0.0.1",
                confirm=True,
            )

        assert result["success"] is True
        mgr.update_traffic_route.assert_awaited_once_with(
            "route-001", domains=new_domains, regions=["US"], next_hop="10.0.0.1"
        )

    @pytest.mark.asyncio
    async def test_enabled_only_still_works(self):
        mgr = _mock_manager()
        with patch("unifi_network_mcp.tools.traffic_routes.traffic_route_manager", mgr):
            from unifi_network_mcp.tools.traffic_routes import update_traffic_route

            result = await update_traffic_route("route-001", enabled=False, confirm=True)

        assert result["success"] is True
        mgr.update_traffic_route.assert_awaited_once_with("route-001", enabled=False)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestUpdateTrafficRouteValidation:
    @pytest.mark.asyncio
    async def test_no_fields_provided_errors(self):
        mgr = _mock_manager()
        with patch("unifi_network_mcp.tools.traffic_routes.traffic_route_manager", mgr):
            from unifi_network_mcp.tools.traffic_routes import update_traffic_route

            result = await update_traffic_route("route-001", confirm=True)

        assert result["success"] is False
        assert "At least one updatable field" in result["error"]
        # Should fail fast without touching the controller.
        mgr.get_traffic_route_details.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_target_devices_must_be_list(self):
        mgr = _mock_manager()
        with patch("unifi_network_mcp.tools.traffic_routes.traffic_route_manager", mgr):
            from unifi_network_mcp.tools.traffic_routes import update_traffic_route

            result = await update_traffic_route("route-001", target_devices={"type": "ALL_CLIENTS"}, confirm=True)

        assert result["success"] is False
        assert "must be a list" in result["error"]
        mgr.update_traffic_route.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_target_devices_entry_requires_type(self):
        mgr = _mock_manager()
        with patch("unifi_network_mcp.tools.traffic_routes.traffic_route_manager", mgr):
            from unifi_network_mcp.tools.traffic_routes import update_traffic_route

            result = await update_traffic_route(
                "route-001", target_devices=[{"client_mac": "aa:bb:cc:dd:ee:ff"}], confirm=True
            )

        assert result["success"] is False
        assert "must be an object with a 'type'" in result["error"]
        mgr.update_traffic_route.assert_not_awaited()
