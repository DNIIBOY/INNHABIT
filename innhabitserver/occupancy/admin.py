from django.contrib import admin

from . import models

admin.site.register(models.Entrance)
admin.site.register(models.EntryEvent)
admin.site.register(models.ExitEvent)
