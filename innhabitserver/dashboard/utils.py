from dataclasses import dataclass

from django.http import HttpRequest


@dataclass
class FakeMetadata:
    request: HttpRequest
