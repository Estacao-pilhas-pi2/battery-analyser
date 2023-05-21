import os
import enum
import random

import numpy as np
import tensorflow as tf
from keras.models import load_model
import keras.utils as kutils
from typing import Dict,Tuple
from pathlib import Path

MODEL = load_model(Path(os.path.dirname(__file__)) / "model")
IMAGE_SIZE = (256, 256)


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
    D = 3
    UNKNOWN = 4


def predict(image_path: str) -> Tuple[Battery, float]:
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
    battery_image = _read_image(image_path)
    prediction = MODEL.predict(battery_image)
    # return Battery(random.randint(0, len(Battery) - 1)), '0.5'
    return _format_prediction(prediction)


def _read_image(image_path):
    battery_image = kutils.load_img(image_path, target_size=IMAGE_SIZE)
    battery_image = kutils.img_to_array(battery_image)
    battery_image = tf.expand_dims(battery_image, 0)
    battery_image /= 255
    return battery_image


def _format_prediction(prediction):
    score = tf.nn.softmax(prediction[0])
    result_indice = np.argmax(score)
    category = Battery(result_indice)
    percentage = np.max(score) * 100
    return category, f'{percentage:.2f}'
