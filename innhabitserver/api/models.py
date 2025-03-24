from django.contrib.auth.hashers import check_password, make_password
from django.db import models
from django.utils.crypto import get_random_string
from occupancy.models import Device


class APIKeyManager(models.Manager):
    def create(self, device: Device):
        prefix = get_random_string(length=16)
        key = get_random_string(length=64)
        api_key = f"inn-{prefix}-{key}"
        super().create(device=device, prefix=prefix, key=make_password(api_key))
        return api_key

    def get_from_key(self, key: str) -> "DeviceAPIKey | None":
        parts = key.split("-")
        if len(parts) != 3:
            return None
        prefix = parts[1]
        obj = self.filter(prefix=prefix).first()
        if not obj:
            return None
        if check_password(key, obj.key):
            return obj
        return None


class DeviceAPIKey(models.Model):
    device = models.OneToOneField(Device, on_delete=models.CASCADE)
    prefix = models.CharField(max_length=16, unique=True, editable=False)
    key = models.CharField(max_length=128, editable=False)
    created_at = models.DateTimeField(auto_now_add=True)

    objects = APIKeyManager()
