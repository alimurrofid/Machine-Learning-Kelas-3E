# LANGKAH 1 - Load Image
from PIL import Image

img = Image.open('Jobsheet 2/dataset/Lenna_(test_image).png')
img.show() # menampilkan gambar

'''
the show() method is used to display the image in the default image viewer associated with your operating system.
'''


# LANGKAH 2 - Ekstrak Fitur
# Ekstrak setiap channel red, green, blue
r, g, b = img.split()

# Cek panjang ukuran channel red
print(len(r.histogram()))

# Cetak fitur histogram pada channel red
print(r.histogram())