from django import forms

class ImageForm(forms.Form):
    imagefile = forms.ImageField(label='Select a Close image of the plant focussing the Leaves')
