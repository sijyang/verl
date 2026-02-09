class SleepLevel:
    RELEASE_KV_CACHE_ONLY = 1  # Release KV cache only
    RELEASE_ALL = 2            # Release KV cache and model weights


class ATOMDefaults:
    SLEEP_LEVEL = SleepLevel.RELEASE_ALL
    TEMPERATURE = 1.0
