REF_AUDIO_PATH: str = r"C:\Code\Github\Project-General-Kalani\models\voice_samples\voice_sample3_cleared.wav"
REF_TEXT: str = """
    They've managed to penetrate our shields and are using the cover of the highlands against our forces. 
    To defeat them will take time. Then, we are lost. Do not underestimate our means if the rebel army falls, 
    the citizens will loose their courage.
"""

VERSION: dict[str, str | dict[str, float]] = {
    "name": "battle_worn_general",
    "description": "Battle-worn authority - experienced but damaged",
    "params": {
        "temperature": 0.55,
        "top_p": 0.85,
        "speed": 0.90,
        "repetition_penalty": 1.1
    }
}