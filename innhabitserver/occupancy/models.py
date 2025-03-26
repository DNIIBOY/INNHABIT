from django.db import models


class Entrance(models.Model):
    name = models.CharField(max_length=64)

    def __str__(self) -> str:
        return f"Entrance: {self.name}"


class Device(models.Model):
    entrance = models.OneToOneField(Entrance, on_delete=models.CASCADE)

    def __str__(self) -> str:
        return f"Device: {self.entrance}"


class DeviceImage(models.Model):
    device = models.ForeignKey(
        Device,
        on_delete=models.CASCADE,
        related_name="images",
    )
    image = models.ImageField(upload_to="device_images/")
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self) -> str:
        return f"DeviceImage: {self.device} ({self.created_at})"


class Event(models.Model):
    class Meta:
        abstract = True

    timestamp = models.DateTimeField()
    entrance = models.ForeignKey(Entrance, on_delete=models.CASCADE)


class EntryEvent(Event):
    entrance = models.ForeignKey(
        Entrance, on_delete=models.CASCADE, related_name="entries"
    )

    def __str__(self) -> str:
        return f"EntryEvent: {self.entrance.name} ({self.timestamp})"


class ExitEvent(Event):
    entrance = models.ForeignKey(
        Entrance, on_delete=models.CASCADE, related_name="exits"
    )

    def __str__(self) -> str:
        return f"ExitEvent: {self.entrance.name} ({self.timestamp})"
