from rest_framework.permissions import BasePermission
from rest_framework.request import Request
from rest_framework.views import View

from .models import DeviceAPIKey


class DeviceAPIKeyPermission(BasePermission):
    prefix = "Bearer"

    def has_permission(self, request: Request, view: View) -> bool:
        auth = request.headers.get("Authorization")
        if not auth or not auth.startswith(f"{self.prefix} "):
            return False
        api_key = auth[len(self.prefix) + 1 :]
        request.auth = DeviceAPIKey.objects.get_from_key(api_key)
        return bool(request.auth)
