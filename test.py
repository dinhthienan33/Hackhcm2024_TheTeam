import easyocr
def perform_ocr(image):
    ocr_reader = easyocr.Reader(['en'])
    result = ocr_reader.readtext(np.array(image))
    ocr_texts = [line[1] for line in result]
    return ocr_texts
img='BZ1A2269.jpg'
print(perform_ocr(img))