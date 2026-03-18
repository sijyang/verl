from enum import IntEnum


class SleepLevel(IntEnum):
    RELEASE_KV_CACHE_ONLY = 1  # Release KV cache only
    RELEASE_ALL = 2            # Release KV cache and model weights


class ATOMDefaults:
    SLEEP_LEVEL = SleepLevel.RELEASE_ALL
    LAYERED_SUMMON_SLEEP_LEVEL = SleepLevel.RELEASE_KV_CACHE_ONLY
    TEMPERATURE = 1.0
    BATCH_TIMEOUT = 0.5  # Seconds to wait before dispatching an incomplete batch


class IPCConfig:
    DEFAULT_BUCKET_SIZE_MB = 2048  # Default bucket size in MB for IPC weight transfer
