"""Fresh-read guarantees for safety-critical NetworkManager callers."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from unifi_core.network.managers.network_manager import NetworkManager


@pytest.mark.asyncio
async def test_get_networks_force_refresh_bypasses_the_shared_cache() -> None:
    connection = MagicMock()
    connection.site = "default"
    connection.get_cached = MagicMock(return_value=[{"_id": "cached-network", "purpose": "vpn"}])
    connection.request = AsyncMock(return_value={"data": [{"_id": "fresh-wan", "purpose": "wan"}]})

    networks = await NetworkManager(connection).get_networks(force_refresh=True)

    assert networks == [{"_id": "fresh-wan", "purpose": "wan"}]
    connection.get_cached.assert_not_called()
    connection.request.assert_awaited_once()
    connection._update_cache.assert_called_once()


@pytest.mark.asyncio
async def test_get_network_details_forwards_force_refresh_to_network_list() -> None:
    manager = NetworkManager(MagicMock())
    manager.get_networks = AsyncMock(return_value=[{"_id": "wan-target", "purpose": "wan"}])

    network = await manager.get_network_details("wan-target", force_refresh=True)

    assert network["purpose"] == "wan"
    manager.get_networks.assert_awaited_once_with(force_refresh=True)
