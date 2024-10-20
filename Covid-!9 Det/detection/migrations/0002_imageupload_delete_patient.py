# Generated by Django 5.0.7 on 2024-07-22 16:08

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('detection', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='ImageUpload',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('image', models.ImageField(upload_to='uploads/')),
                ('result', models.CharField(blank=True, max_length=10, null=True)),
                ('uploaded_at', models.DateTimeField(auto_now_add=True)),
            ],
        ),
        migrations.DeleteModel(
            name='Patient',
        ),
    ]
