from django.db import models


class LabelledDate(models.Model):
    date = models.DateField(unique=True)
    label = models.CharField(max_length=100)

    def __str__(self) -> str:
        return f"{self.label} - {self.date}"
