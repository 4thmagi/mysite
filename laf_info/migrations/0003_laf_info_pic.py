# Generated by Django 2.0.3 on 2018-05-24 11:33

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('laf_info', '0002_auto_20180524_1516'),
    ]

    operations = [
        migrations.AddField(
            model_name='laf_info',
            name='pic',
            field=models.ImageField(blank=True, null=True, upload_to='pic'),
        ),
    ]
