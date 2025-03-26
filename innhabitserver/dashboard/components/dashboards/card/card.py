from django_components import Component, register


@register("card")
class Card(Component):
    template_name = "card.html"
