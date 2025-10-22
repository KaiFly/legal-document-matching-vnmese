from app.legal_identity_extract import LegalIdentity


def detect_text(path):
    """Detects text in the file."""
    from google.cloud import vision

    client = vision.ImageAnnotatorClient()

    with open(path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations
    print("Texts:")
    return texts[0].description

    # for text in texts:
    #     print(f'\n"{text.description}"')
    #
    #     vertices = [
    #         f"({vertex.x},{vertex.y})" for vertex in text.bounding_poly.vertices
    #     ]
    #
    #     print("bounds: {}".format(",".join(vertices)))

    if response.error.message:
        raise Exception(
            "{}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors".format(response.error.message)
        )

def detect_text_from_binary(bytes_data):
    """Detects text in the file."""
    from google.cloud import vision

    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=bytes_data)

    response = client.text_detection(image=image)
    texts = response.text_annotations
    return texts[0].description

result = detect_text('/Users/thuyenhx/Downloads/cccd_test4.jpg')
# result = detect_text_from_binary('/Users/thuyenhx/Downloads/cccd_test4.jpg')
identity = LegalIdentity(result)

# Print extracted information
print(f"ID Number: {identity.id_number}")
print(f"Full Name: {identity.full_name}")
print(f"Date of Birth: {identity.dob}")
print(f"Sex: {identity.gender}")
print(f"Nationality: {identity.nationality}")
print(f"Place of Origin: {identity.place_of_origin}")
print(f"Place of Residence: {identity.place_of_residence}")