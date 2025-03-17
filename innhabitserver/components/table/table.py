from django_components import Component, register


@register("table")
class Table(Component):
    template_name = "table.html"
