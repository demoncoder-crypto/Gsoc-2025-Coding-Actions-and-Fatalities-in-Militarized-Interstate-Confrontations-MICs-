import os

class Config:
    # Data paths
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Dataset")
    ARTICLES_DIR = os.path.join(DATA_DIR, "New York Times")
    STATE_SYSTEM_FILE = os.path.join(DATA_DIR, "states2016.csv")
    
    # Output paths
    OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
    MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
    RESULTS_FILE = os.path.join(OUTPUT_DIR, "mic_results.csv")
    
    # Create output directories if they don't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Model parameters
    DOCUMENT_CLASSIFIER_MODEL = "distilroberta-base"
    NER_MODEL = "distilroberta-base"
    EVENT_EXTRACTION_MODEL = "distilroberta-base"
    RELATION_MODEL = "distilroberta-base"
    
    # Training parameters
    BATCH_SIZE = 8
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 3
    MAX_SEQ_LENGTH = 512
    
    # Preprocessing parameters
    MIN_ARTICLE_LENGTH = 100
    MAX_ARTICLE_LENGTH = 2000
    
    # Analysis parameters
    YEARS_TO_ANALYZE = list(range(2015, 2024))  # 2015-2023 as specified in the task
    
    # Keywords for filtering
    DEATH_KEYWORDS = [
        "killed", "died", "dead", "death", "fatality", "fatalities", "casualty", "casualties",
        "perished", "loss of life", "lost lives", "lives lost"
    ]
    
    MILITARY_KEYWORDS = [
        "soldier", "troop", "military", "army", "navy", "air force", "marine", 
        "armed forces", "personnel", "serviceman", "servicewoman", "servicemen", 
        "servicewomen", "airman", "airmen", "sailor", "combat", "fighter", "officer"
    ] 