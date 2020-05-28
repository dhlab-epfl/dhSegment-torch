from enum import Enum


class PredictionType(Enum):
    """

    :cvar CLASSIFICATION:
    :cvar REGRESSION:
    :cvar MULTILABEL:
    """

    CLASSIFICATION = "CLASSIFICATION"
    REGRESSION = "REGRESSION"
    MULTILABEL = "MULTILABEL"

    @classmethod
    def parse(cls, prediction_type: str) -> "PredictionType":
        if prediction_type == "CLASSIFICATION":
            return PredictionType.CLASSIFICATION
        elif prediction_type == "REGRESSION":
            return PredictionType.REGRESSION
        elif prediction_type == "MULTILABEL":
            return PredictionType.MULTILABEL
        else:
            raise NotImplementedError(
                "Unknown prediction type : {}".format(prediction_type)
            )
