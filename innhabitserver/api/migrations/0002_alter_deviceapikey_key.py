# Generated by Django 5.1.7 on 2025-03-25 10:08

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("api", "0001_initial"),
    ]

    operations = [
        migrations.AlterField(
            model_name="deviceapikey",
            name="key",
            field=models.CharField(editable=False, max_length=256),
        ),
    ]
