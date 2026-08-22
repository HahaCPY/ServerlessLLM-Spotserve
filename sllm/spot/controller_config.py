from typing import Mapping, Optional


LEGACY_MIGRATION_WARNING = (
    "enable_migration is a deprecated legacy MigrationRouter flag and is "
    "ignored by the SpotServe controller. Configure SpotServe migration via "
    "model router_config: enable_context_migration, enable_reparallelization, "
    "and recovery_policy=stateful_recovery."
)


def spotserve_scheduler_config(config: Optional[Mapping]) -> tuple[dict, bool]:
    controller_config = dict(config or {})
    scheduler_config = dict(
        controller_config.get("scheduler_config", {}) or {}
    )
    legacy_requested = bool(
        controller_config.get("enable_migration", False)
        or scheduler_config.get("enable_migration", False)
    )
    scheduler_config["enable_migration"] = False
    return scheduler_config, legacy_requested
