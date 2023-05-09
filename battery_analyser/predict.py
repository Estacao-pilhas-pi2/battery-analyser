import enum
import random

import numpy as np
from keras.models import load_model
from keras.preprocessing import image

# MODEL = load_model('model')
IMAGE_SIZE = (64, 64)


class Battery(enum.Enum):
    """
    Classificação das baterias

    Attributes:

        V9: Bateria 9V
        AA: Pilha AA
        AAA: Pilha AAA
        C: Pilha C
        D: Pilha D
        UNKNOWN: Objeto desconhecido
    """

    V9 = 0
    AA = 1
    AAA = 2
    C = 3
    D = 4
    UNKNOWN = 5


def predict(image_path: str) -> tuple[Battery, float]:
    """
    Classifica uma imagem de uma bateria

    Args:
        image_path: Caminho para a imagem da bateria

    Returns:
        Uma tupla com a classificação da bateria e a porcentagem da predição.

    Examples:
        >>> predict('caminho_bateria_tipo_aaa.png')
        (Battery.AAA, "0.5")
        >>> predict('foto_aleatoria.jpg')
        (Battery.UNKNOWN, "0.5")
    """
    # battery_image = _read_image(image_path)
    # prediction = MODEL.predict(battery_image)
    # return _format_prediction(prediction)
    return Battery(random.randint(0, len(Battery) - 1)), '0.5'


def _read_image(image_path):
    battery_image = image.load_img(image_path, target_size=IMAGE_SIZE)
    battery_image = image.img_to_array(battery_image)
    battery_image = np.expand_dims(battery_image, axis=0)
    battery_image /= 255
    return battery_image


def _format_prediction(prediction):
    result_indice = np.argmax(prediction[0])
    category = Battery(result_indice)
    percentage = prediction[0][result_indice] * 100
    return category, f'{percentage:.2f}'
