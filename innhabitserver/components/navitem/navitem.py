from django_components import Component, register


@register("navitem")
class Navitem(Component):
    template_name = "navitem.html"

    def get_context_data(self, active: bool = False) -> dict:
        return {"active": active}
