from enum import Enum


class QuestionType(str, Enum):
    MCQ = "MCQ"
    TOF = "TOF"
    FIB = "FIB"
    MTF = "MTF"
    DES = "DES"


class Difficulty(str, Enum):
    VERY_EASY = "VERY_EASY"
    EASY = "EASY"
    MEDIUM = "MEDIUM"
    HARD = "HARD"
    VERY_HARD = "VERY_HARD"
