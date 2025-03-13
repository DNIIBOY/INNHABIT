from django_components import Component, register


@register("busiest_days")
class BusiestDays(Component):
    template_name = "busiest_days.html"
