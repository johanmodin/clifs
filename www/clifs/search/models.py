from django.db import models

# Create your models here.
class Image(models.Model):
    image_data = models.ImageField()
    title = models.CharField(max_length=256)
    time = models.FloatField()
    match_score = models.FloatField()


class Query(models.Model):
    query = models.CharField(max_length=256)

    def __str__(self):
        return self.query
