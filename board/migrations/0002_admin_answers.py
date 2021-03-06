# Generated by Django 3.1 on 2020-08-25 07:12

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('board', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Admin',
            fields=[
                ('admin_id', models.CharField(max_length=45, primary_key=True, serialize=False)),
                ('admin_pw', models.CharField(max_length=45)),
            ],
            options={
                'db_table': 'admin',
                'managed': False,
            },
        ),
        migrations.CreateModel(
            name='Answers',
            fields=[
                ('board_no', models.IntegerField()),
                ('answer_no', models.IntegerField(primary_key=True, serialize=False)),
                ('answer', models.CharField(blank=True, max_length=225, null=True)),
                ('article_title', models.CharField(blank=True, max_length=225, null=True)),
                ('article_url', models.CharField(blank=True, max_length=225, null=True)),
                ('good', models.IntegerField(blank=True, null=True)),
                ('bad', models.IntegerField(blank=True, null=True)),
            ],
            options={
                'db_table': 'answers',
                'managed': False,
            },
        ),
    ]
