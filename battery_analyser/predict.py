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


Predicton = Tuple[Battery, float]


def predict(image: np.ndarray) -> Predicton:
    """
    Classifica uma imagem de uma bateria

    Args:
        image: Array de bytes de 3 dimensões com a imagem

    Returns:
        Uma Predicton com a classificação da bateria e a porcentagem da predição.

    Examples:
        >>> predict('caminho_bateria_tipo_aaa.png')
        (Battery.AAA, "0.5")
        >>> predict('foto_aleatoria.jpg')
        (Battery.UNKNOWN, "0.5")
    """
    image = tf.expand_dims(image, 0)
    image /= 255
    prediction = MODEL.predict(image)
    return _format_prediction(prediction)


def predict_from_path(image_path: str) -> Predicton:
    """
    Classifica uma imagem de uma bateria

    Args:
        image_path: Caminho para a imagem da bateria

    Returns:
        Uma Predicton com a classificação da bateria e a porcentagem da predição.

    Examples:
        >>> predict('caminho_bateria_tipo_aaa.png')
        (Battery.AAA, "0.5")
        >>> predict('foto_aleatoria.jpg')
        (Battery.UNKNOWN, "0.5")
    """
    battery_image = _read_image(image_path)
    return predict(battery_image)


def _read_image(image_path):
    battery_image = kutils.load_img(image_path, target_size=IMAGE_SIZE)
    battery_image = kutils.img_to_array(battery_image)
    return battery_image


def _format_prediction(prediction):
    score = tf.nn.softmax(prediction[0])
    result_indice = np.argmax(score)
    category = Battery(result_indice)
    percentage = np.max(score) * 100
    return category, f'{percentage:.2f}%'
