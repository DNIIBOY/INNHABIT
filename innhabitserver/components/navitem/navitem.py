from django_components import Component, register


@register("navitem")
class Navitem(Component):
    template_name = "navitem.html"
