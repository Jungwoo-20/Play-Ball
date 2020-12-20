from django.db import models


# Create your models here.


class Board(models.Model):
    no = models.AutoField(db_column='No', primary_key=True)  # Field name made lowercase.
    question = models.CharField(max_length=45, blank=True, null=True)
    writer = models.CharField(max_length=45, blank=True, null=True)
    date = models.CharField(max_length=45, blank=True, null=True)
    answer = models.IntegerField(blank=True, null=True)
    pw = models.CharField(max_length=45, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'board'


class Answers(models.Model):
    board_no = models.IntegerField()
    answer_no = models.IntegerField(primary_key=True)
    answer = models.CharField(max_length=225, blank=True, null=True)
    article_title = models.CharField(max_length=225, blank=True, null=True)
    article_url = models.CharField(max_length=225, blank=True, null=True)
    good = models.IntegerField(blank=True, null=True)
    bad = models.IntegerField(blank=True, null=True)
    date = models.DateTimeField(blank=True, null=True)
    reinforce = models.IntegerField(blank=True, null=True)
    ratio = models.IntegerField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'answers'
        unique_together = (('answer_no', 'board_no'),)


class Admin(models.Model):
    admin_id = models.CharField(primary_key=True, max_length=45)
    admin_pw = models.CharField(max_length=45)

    class Meta:
        managed = False
        db_table = 'admin'


class ModelScore(models.Model):
    f1_score = models.FloatField(primary_key=True)

    class Meta:
        managed = False
        db_table = 'model_score'


class Rank(models.Model):
    no = models.AutoField(primary_key=True)
    title = models.CharField(max_length=45, blank=True, null=True)
    rank_no = models.IntegerField(blank=True, null=True)
    name = models.CharField(max_length=45, blank=True, null=True)
    team = models.CharField(max_length=45, blank=True, null=True)
    info = models.CharField(max_length=45, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'rank'
