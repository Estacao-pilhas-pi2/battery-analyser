from PIL import Image
import sys

image = Image.open(sys.argv[1])

image.crop((0,60, 430, 420)).save(sys.argv[1])
print("Cropped:", sys.argv[1])
