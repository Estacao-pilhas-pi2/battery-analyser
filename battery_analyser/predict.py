import os
import enum

from pathlib import Path

import numpy as np

from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision


MODEL = str(Path(os.path.dirname(__file__)) / "model.tflite")
NUM_THREADS = int(os.environ.get("ANALYSER_NUM_THREADS", "2"))
WIDTH, HEIGHT = IMAGE_SIZE = (256, 256)


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


base_options = core.BaseOptions(
    file_name=MODEL, num_threads=NUM_THREADS)

classification_options = processor.ClassificationOptions(max_results=1)

options = vision.ImageClassifierOptions(
    base_options=base_options, classification_options=classification_options)

classifier = vision.ImageClassifier.create_from_options(options)


def predict(image: np.ndarray) -> Battery:
    """
    Classifica uma imagem de uma bateria

    Args:
        image: Array de bytes de 3 dimensões com a imagem

    Returns:
        Uma Predicton com a classificação da bateria e a porcentagem da predição.
    """
    tensor_image = vision.TensorImage.create_from_array(image)
    categories = classifier.classify(tensor_image)
    if not categories.classifications or not categories.classifications[0].categories:
        return None
    return Battery(categories.classifications[0].categories[0].index)
