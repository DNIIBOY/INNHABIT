import re

from django.contrib.auth.hashers import check_password, make_password
from django.db import models
from django.utils.crypto import get_random_string
from occupancy.models import Device

KEY_PATTERN = re.compile(r"^(inn)-(\w{16})-(\w{64})$")


class APIKeyManager(models.Manager):
    def create(self, device: Device) -> str:
        prefix = get_random_string(length=16)
        key = get_random_string(length=64)
        api_key = f"inn-{prefix}-{key}"
        hashed = make_password(api_key, hasher="sha512")
        super().create(device=device, prefix=prefix, key=hashed)
        return api_key

    def get_from_key(self, key: str) -> "DeviceAPIKey | None":
        match = KEY_PATTERN.match(key)
        if not match:
            return None
        prefix = match.group(2)
        obj = self.filter(prefix=prefix).first()
        if not obj:
            return None
        if check_password(key, obj.key, preferred="sha512"):
            return obj
        return None


class DeviceAPIKey(models.Model):
    device = models.OneToOneField(
        Device,
        on_delete=models.CASCADE,
        related_name="api_key",
    )
    prefix = models.CharField(max_length=16, unique=True, editable=False)
    key = models.CharField(max_length=256, editable=False)
    created_at = models.DateTimeField(auto_now_add=True)

    objects = APIKeyManager()

    def __str__(self) -> str:
        return f"inn-{self.prefix}-{'*' * 8}"
