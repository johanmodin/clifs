from django import forms

class QueryForm(forms.Form):
    query = forms.CharField(widget=forms.TextInput(attrs={'size': '40',
                'placeholder': 'A search query, e.g., "A yellow ambulance"'}))
