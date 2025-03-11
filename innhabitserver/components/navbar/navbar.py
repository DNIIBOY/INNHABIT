from django_components import Component, register


@register("navbar")
class ProductCard(Component):
    template_name = "navbar.html"
