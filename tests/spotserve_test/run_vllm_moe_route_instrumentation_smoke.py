"""Wrapper for the real vLLM MoE route instrumentation smoke."""

import asyncio

from sllm.spot.moe_route_instrumentation_smoke import main


if __name__ == "__main__":
    asyncio.run(main())
