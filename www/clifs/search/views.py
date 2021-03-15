from django.shortcuts import render
from django.http import HttpResponse

import requests

#from .models import Videos
from .models import Query, Image
from .forms import QueryForm

SEARCH_URL = 'http://search-engine:5000/search/{}'

def search(request):
    form = QueryForm()
    images = []
    if request.method == 'POST':
        query = Query(query = request.POST['query'])
        images_metadata = requests.get(SEARCH_URL.format(query)).json()
        for image_metadata in images_metadata:
            img_obj, _ = Image.objects.get_or_create(
                image_data='results/{}'.format(image_metadata['image_path']),
                title=image_metadata['image_path'],
                time=float(image_metadata['time']),
                match_score=float(image_metadata['score'])
                )
            images.append(img_obj)

    return render(request, 'search.html', {'nbar': 'search',
                                           'images': images,
                                           'form': form})
