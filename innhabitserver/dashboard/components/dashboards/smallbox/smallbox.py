from django_components import Component, register


@register("smallbox")
class SmallBox(Component):
    template_name = "smallbox.html"
