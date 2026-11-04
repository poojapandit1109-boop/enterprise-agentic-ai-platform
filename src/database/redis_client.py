
import redis

from src.config import get_settings


settings = get_settings()

redis_client = redis.Redis.from_url(
    settings.redis_url,
    decode_responses=True,
    socket_connect_timeout=5,
    socket_timeout=5,
)


def check_redis_connection() -> bool:
    """Check whether Redis is reachable."""
    try:
        return bool(redis_client.ping())
    except redis.RedisError:
        return False
