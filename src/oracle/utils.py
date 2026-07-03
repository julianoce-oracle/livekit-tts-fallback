import logging

from oci.config import validate_config
from oci.regions import REGIONS

logger = logging.getLogger("livekit.plugins.oracle")


def validate_and_prepare_config(config_original: dict, region: str):
    config = config_original.copy()
    if region not in REGIONS:
        raise Exception("Invalid region")
    config["region"] = region
    validate_config(config)
    return config
