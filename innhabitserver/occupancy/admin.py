from django.contrib import admin

from . import models

admin.site.register(models.Device)
admin.site.register(models.DeviceImage)
admin.site.register(models.DeviceSettings)
admin.site.register(models.Entrance)
admin.site.register(models.EntryEvent)
admin.site.register(models.ExitEvent)
admin.site.register(models.TestEntryEvent)
admin.site.register(models.TestExitEvent)
