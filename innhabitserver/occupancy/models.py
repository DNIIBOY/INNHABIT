from django.db import models


class Entrance(models.Model):
    name = models.CharField(max_length=64)

    def __str__(self) -> str:
        return f"Entrance: {self.name}"


class Event(models.Model):
    class Meta:
        abstract = True

    timestamp = models.DateTimeField()
    entrance = models.ForeignKey(Entrance, on_delete=models.CASCADE)


class EntryEvent(Event):
    entrance = models.ForeignKey(
        Entrance, on_delete=models.CASCADE, related_name="entries"
    )


class ExitEvent(Event):
    entrance = models.ForeignKey(
        Entrance, on_delete=models.CASCADE, related_name="exits"
    )
