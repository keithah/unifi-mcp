"""Safety validation for TrafficRouteManager mutations."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from unifi_core.network.managers.traffic_route_manager import TrafficRouteManager

VALID_TARGET = [{"type": "CLIENT", "client_mac": "aa:bb:cc:dd:ee:ff"}]


def _manager(*, purpose: str = "wan") -> tuple[TrafficRouteManager, MagicMock, MagicMock]:
    connection = MagicMock()
    connection.site = "default"
    connection.request = AsyncMock(return_value={"data": {"_id": "route-new"}})
    network_manager = MagicMock()
    network_manager.get_network_details = AsyncMock(return_value={"_id": "wan-target", "purpose": purpose})
    manager = TrafficRouteManager(connection, network_manager=network_manager)
    return manager, connection, network_manager


@pytest.mark.asyncio
async def test_create_rejects_internet_route_with_non_wan_target() -> None:
    manager, connection, network_manager = _manager(purpose="remote-user-vpn")

    with pytest.raises(ValueError, match="WAN network"):
        await manager.create_traffic_route(
            {
                "description": "Desktop route",
                "matching_target": "INTERNET",
                "network_id": "vpn-target",
                "target_devices": VALID_TARGET,
                "enabled": True,
            }
        )

    network_manager.get_network_details.assert_awaited_once_with("vpn-target", force_refresh=True)
    connection.request.assert_not_awaited()


@pytest.mark.asyncio
async def test_create_rejects_disabled_internet_route_with_non_wan_target() -> None:
    manager, connection, _ = _manager(purpose="remote-user-vpn")

    with pytest.raises(ValueError, match="WAN network"):
        await manager.create_traffic_route(
            {
                "description": "Disabled unsafe route",
                "matching_target": "INTERNET",
                "network_id": "vpn-target",
                "target_devices": VALID_TARGET,
                "enabled": False,
            }
        )

    connection.request.assert_not_awaited()


@pytest.mark.asyncio
async def test_update_cannot_enable_unsafe_legacy_internet_route() -> None:
    manager, connection, _ = _manager()
    manager.get_traffic_route_details = AsyncMock(
        return_value={
            "_id": "route-unsafe",
            "description": "Unsafe legacy route",
            "matching_target": "INTERNET",
            "network_id": "vpn-target",
            "target_devices": [{"type": "ALL_CLIENTS"}],
            "enabled": False,
        }
    )

    with pytest.raises(ValueError, match="exactly one explicit CLIENT"):
        await manager.update_traffic_route("route-unsafe", enabled=True)

    connection.request.assert_not_awaited()


@pytest.mark.asyncio
async def test_toggle_cannot_enable_unsafe_legacy_internet_route() -> None:
    manager, connection, _ = _manager()
    manager.get_traffic_route_details = AsyncMock(
        return_value={
            "_id": "route-unsafe",
            "description": "Unsafe legacy route",
            "matching_target": "INTERNET",
            "network_id": "vpn-target",
            "target_devices": [{"type": "ALL_CLIENTS"}],
            "enabled": False,
        }
    )

    with pytest.raises(ValueError, match="exactly one explicit CLIENT"):
        await manager.toggle_traffic_route("route-unsafe")

    connection.request.assert_not_awaited()


@pytest.mark.asyncio
async def test_toggle_revalidates_the_route_refetched_for_mutation() -> None:
    manager, connection, _ = _manager()
    safe_disabled_route = {
        "_id": "route-race",
        "description": "Initially safe route",
        "matching_target": "INTERNET",
        "network_id": "wan-target",
        "target_devices": VALID_TARGET,
        "enabled": False,
    }
    unsafe_disabled_route = {
        **safe_disabled_route,
        "network_id": "vpn-target",
        "target_devices": [{"type": "ALL_CLIENTS"}],
    }
    manager.get_traffic_route_details = AsyncMock(side_effect=[safe_disabled_route, unsafe_disabled_route])

    with pytest.raises(ValueError, match="exactly one explicit CLIENT"):
        await manager.toggle_traffic_route("route-race")

    connection.request.assert_not_awaited()


@pytest.mark.asyncio
async def test_update_ignores_cached_route_before_write() -> None:
    manager, connection, network_manager = _manager()
    stale_cached_route = {
        "_id": "route-fresh",
        "description": "Stale unsafe route",
        "matching_target": "INTERNET",
        "network_id": "vpn-target",
        "target_devices": [{"type": "ALL_CLIENTS"}],
        "enabled": True,
    }
    fresh_controller_route = {
        "_id": "route-fresh",
        "description": "Fresh disabled route",
        "matching_target": "INTERNET",
        "network_id": "wan-target",
        "target_devices": VALID_TARGET,
        "enabled": False,
    }
    connection.get_cached = MagicMock(return_value=[stale_cached_route])
    connection.request = AsyncMock(side_effect=[{"data": [fresh_controller_route]}, {}])

    updated = await manager.update_traffic_route("route-fresh", enabled=False)

    assert updated is True
    connection.get_cached.assert_not_called()
    assert connection.request.await_count == 2
    get_request, put_request = [call.args[0] for call in connection.request.await_args_list]
    assert get_request.method == "get"
    assert get_request.path == "/trafficroutes"
    assert put_request.method == "put"
    assert put_request.data["description"] == "Fresh disabled route"
    network_manager.get_network_details.assert_not_awaited()


@pytest.mark.asyncio
async def test_update_cannot_make_disabled_internet_route_unsafe() -> None:
    manager, connection, _ = _manager(purpose="remote-user-vpn")
    manager.get_traffic_route_details = AsyncMock(
        return_value={
            "_id": "route-disabled",
            "description": "Safe disabled route",
            "matching_target": "INTERNET",
            "network_id": "wan-target",
            "target_devices": VALID_TARGET,
            "enabled": False,
        }
    )

    with pytest.raises(ValueError, match="WAN network"):
        await manager.update_traffic_route("route-disabled", network_id="vpn-target")

    connection.request.assert_not_awaited()


@pytest.mark.asyncio
async def test_update_can_disable_unsafe_legacy_internet_route() -> None:
    manager, connection, network_manager = _manager()
    manager.get_traffic_route_details = AsyncMock(
        return_value={
            "_id": "route-unsafe",
            "description": "Unsafe legacy route",
            "matching_target": "INTERNET",
            "network_id": "vpn-target",
            "target_devices": [{"type": "ALL_CLIENTS"}],
            "enabled": True,
        }
    )

    updated = await manager.update_traffic_route("route-unsafe", enabled=False)

    assert updated is True
    network_manager.get_network_details.assert_not_awaited()
    connection.request.assert_awaited_once()


@pytest.mark.asyncio
async def test_update_kill_switch_rejects_unsafe_enabled_internet_route() -> None:
    manager, connection, _ = _manager()
    manager.get_traffic_route_details = AsyncMock(
        return_value={
            "_id": "route-unsafe",
            "description": "Unsafe legacy route",
            "matching_target": "INTERNET",
            "network_id": "vpn-target",
            "target_devices": [{"type": "ALL_CLIENTS"}],
            "enabled": True,
        }
    )

    with pytest.raises(ValueError, match="exactly one explicit CLIENT"):
        await manager.update_kill_switch("route-unsafe", enabled=True)

    connection.request.assert_not_awaited()


@pytest.mark.asyncio
async def test_create_allows_valid_single_client_wan_route() -> None:
    manager, connection, network_manager = _manager()
    payload = {
        "description": "Desktop route",
        "matching_target": "INTERNET",
        "network_id": "wan-target",
        "target_devices": VALID_TARGET,
        "enabled": True,
    }

    created = await manager.create_traffic_route(payload)

    assert created == {"_id": "route-new"}
    network_manager.get_network_details.assert_awaited_once_with("wan-target", force_refresh=True)
    connection.request.assert_awaited_once()
