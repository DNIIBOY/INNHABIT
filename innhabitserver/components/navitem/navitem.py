from django.urls import reverse
from django_components import Component, register


@register("navitem")
class Navitem(Component):
    template_name = "navitem.html"

    def get_context_data(
        self, url_name: str = "", content: str = "Nav Item", icon: str = "question"
    ) -> dict:
        url = reverse(url_name) if url_name else ""
        return {
            "active": self.request.path == url,
            "icon": icon,
            "url": url,
            "content": content,
        }
