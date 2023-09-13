from PIL import Image
 
 #kode program di bawah untuk membaca atau mencari sebuah file dengan format(img, png)
img = Image.open('Img/Lenna_(test_image).png')
img.show() 

# Ekstrak setiap channel red, green, blue
r, g, b = img.split()

# Cek panjang ukuran channel red
print(len(r.histogram()))

# Cetak fitur histogram pada channel red
print(r.histogram())