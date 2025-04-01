import cv2
import numpy as np
import sys

sys.path.append(".")

from schemas.labels import (
    ImageLabelSchema,
    LabelSchema,
    LabelCoordinatesSchema,
    ObjectSchema,
    ImageShape,
)


def get_image_with_targets(image_labels: ImageLabelSchema) -> np.ndarray:
    image = cv2.imread(image_labels.img_path)
    for label in image_labels.labels:
        x_min, y_min, x_max, y_max = (
            label.coordinates.min_x,
            label.coordinates.min_y,
            label.coordinates.max_x,
            label.coordinates.max_y,
        )
        text = f"{label.object.name} {str(label.confidence) + '%' if label.confidence is not None else ''} {image_labels.status}"
        color = label.object.bbox_color

        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 5)

        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 5
        )

        # Position the text above the top-left corner of the bounding box
        text_x, text_y = x_min, y_min - 2  # 5 pixels above the box
        if text_y < 0:  # Ensure text is within the image
            text_y = y_min + text_height + 2

        # Put the text on top of the filled rectangle
        cv2.putText(
            image,
            text,
            (text_x, text_y - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

    return image


if __name__ == "__main__":
    img_path = "data/icip/test/data/0001.jpg"

    image_label = ImageLabelSchema(
        img_path=img_path,
        labels=[
            LabelSchema(
                object=ObjectSchema.from_initial_name("Taenia spp. egg"),
                confidence=0.98,
                coordinates=LabelCoordinatesSchema(
                    min_x=50, max_x=400, min_y=30, max_y=300
                ),
                image_shape=ImageShape(width=1280, height=960),
            ),
            LabelSchema(
                object=ObjectSchema.from_initial_name("Trichuris trichiura"),
                coordinates=LabelCoordinatesSchema(
                    min_x=100, max_x=200, min_y=100, max_y=180
                ),
                image_shape=ImageShape(width=1280, height=960),
            ),
        ],
    )
    bbox_image = get_image_with_targets(image_label)

    # Display the result
    cv2.imshow("Image with Bounding Box", bbox_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
