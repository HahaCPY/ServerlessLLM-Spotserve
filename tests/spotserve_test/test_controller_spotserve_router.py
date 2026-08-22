from sllm.spot.controller_config import spotserve_scheduler_config


def test_spotserve_controller_ignores_legacy_enable_migration():
    scheduler_config, legacy_requested = spotserve_scheduler_config(
        {
            "enable_migration": True,
            "scheduler_config": {
                "enable_migration": True,
                "enable_spot_risk_aware": True,
            },
        }
    )

    assert legacy_requested is True
    assert scheduler_config["enable_migration"] is False
    assert scheduler_config["enable_spot_risk_aware"] is True


def test_spotserve_controller_keeps_migration_disabled_by_default():
    scheduler_config, legacy_requested = spotserve_scheduler_config({})

    assert legacy_requested is False
    assert scheduler_config["enable_migration"] is False
