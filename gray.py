import imageio.v2 as img
import numpy as np
import matplotlib.pyplot as plt

path = "C:\\Users\\Administrator\\Downloads\\Bangunan.jpg"
image_source = img.imread(path)
imgG = np.mean(image_source, axis=2).astype(np.uint8)
total_pixels = imgG.size

hist, bins = np.histogram(image_source.flatten(), bins = 256, range = [0,256])
histG, binsG= np.histogram(imgG.flatten(), bins = 256, range = [0,256])

plt.figure(figsize=(9,6))

plt.subplot(2,2,1)
plt.imshow(image_source)
plt.title("Gambar Asli")
plt.axis("off")

plt.subplot(2,2,2)
plt.imshow(imgG, cmap="gray")
plt.title("GraySclae")
plt.axis("off")

plt.subplot(2,2,3)
plt.plot(hist, color= "blue", label='asli')
plt.title("Histogram Citra Asli")
plt.legend()

plt.subplot(2,2,4)
plt.plot(histG, color= "Black", label='Grayscale')
plt.title(f"Histogram Citra Grayscale (Total Piksel: {total_pixels})")
plt.legend()

plt.tight_layout()
plt.show()

