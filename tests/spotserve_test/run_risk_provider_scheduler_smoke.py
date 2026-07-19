"""Executable smoke for provider metadata flowing into FcfsScheduler."""

import asyncio

from sllm.schedulers.fcfs_scheduler import FcfsScheduler


async def main():
    scheduler = FcfsScheduler({"enable_spot_risk_aware": True})
    result = await scheduler._collect_provider_risk_metadata(
        {
            "node-1": {"free_gpu": 1, "total_gpu": 1},
            "node-x": {"free_gpu": 1, "total_gpu": 1},
        }
    )
    real = result["node-1"]
    assert real["spot_risk"] == 0.12
    assert real["risk_metadata_source"] == "environment"
    assert result["node-x"]["risk_metadata_source"] == "conservative"
    assert result["node-x"]["risk_confidence"] == 0.0
    print({"status": "passed", "metadata": result})


if __name__ == "__main__":
    asyncio.run(main())
