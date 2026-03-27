"""
utils.py
--------
Utility functions for the Plant Identification & Benefit Prediction System.
Includes:
  - Image preprocessing for model input
  - Plant information database (JSON-style dictionary)
  - Prediction helper
"""

import numpy as np
from PIL import Image
import json

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

IMAGE_SIZE = (224, 224)   # MobileNetV2 expected input size

# Class names must match the folder names used during training (sorted alphabetically)
CLASS_NAMES = [
    "Aloe_Vera",
    "Banana",
    "Basil",
    "Mango",
    "Neem",
    "Rose",
    "Tulsi",
    "Turmeric",
    "Almond",
    "Papaya",
]

# ──────────────────────────────────────────────────────────────────────────────
# PLANT INFORMATION DATABASE
# ──────────────────────────────────────────────────────────────────────────────

PLANT_INFO: dict = {
    "Aloe_Vera": {
        "common_name": "Aloe Vera",
        "scientific_name": "Aloe barbadensis miller",
        "family": "Asphodelaceae",
        "description": (
            "A succulent plant species widely cultivated for its medicinal gel "
            "stored in thick, fleshy leaves."
        ),
        "benefits": [
            "Soothes sunburns and skin irritation",
            "Rich in antioxidants and vitamins C & E",
            "Supports digestive health",
            "Natural moisturiser",
            "Accelerates wound healing",
        ],
        "medicinal_uses": [
            "Topical treatment for burns, eczema, and psoriasis",
            "Aloe juice used for constipation relief",
            "Mouthwash alternative for gum inflammation",
        ],
        "agricultural_uses": [
            "Natural plant growth stimulant when diluted",
            "Organic seed coating to improve germination",
        ],
        "vrikshayurveda": (
            "Vrikshayurveda texts (Surapala, ~10th c.) recommend aloe paste "
            "for healing tree bark wounds and promoting root growth."
        ),
    },

    "Banana": {
        "common_name": "Banana",
        "scientific_name": "Musa acuminata",
        "family": "Musaceae",
        "description": (
            "One of the world's most consumed fruits, growing on large herbaceous plants."
        ),
        "benefits": [
            "Excellent source of potassium and magnesium",
            "Natural energy booster",
            "Rich in Vitamin B6",
            "Supports heart health",
            "Contains pectin aiding digestion",
        ],
        "medicinal_uses": [
            "Unripe banana used for diarrhoea management",
            "Banana peel applied to minor wounds and insect bites",
            "Flower extract used in traditional diabetes management",
        ],
        "agricultural_uses": [
            "Banana peels used as potassium-rich organic fertiliser",
            "Leaves used as eco-friendly mulch",
        ],
        "vrikshayurveda": (
            "Banana is considered sacred in Ayurvedic tradition; its various parts "
            "— root, stem, flower, and fruit — are all therapeutically valued."
        ),
    },

    "Basil": {
        "common_name": "Basil",
        "scientific_name": "Ocimum basilicum",
        "family": "Lamiaceae",
        "description": (
            "A culinary herb with a strong, pleasant aroma widely used in cooking "
            "and traditional medicine across cultures."
        ),
        "benefits": [
            "Anti-inflammatory properties due to eugenol",
            "Antibacterial and antifungal activity",
            "Rich in Vitamin K",
            "Supports liver health",
            "Natural stress reducer",
        ],
        "medicinal_uses": [
            "Herbal tea for cold, fever, and headaches",
            "Essential oil used in aromatherapy",
            "Poultice for bites and skin infections",
        ],
        "agricultural_uses": [
            "Natural companion plant that repels aphids and whitefly",
            "Improves flavour of neighbouring tomato plants",
        ],
        "vrikshayurveda": (
            "Closely related to Tulsi, Basil shares similar reverence in "
            "Ayurvedic plant sciences; planted near water bodies for purification."
        ),
    },

    "Mango": {
        "common_name": "Mango",
        "scientific_name": "Mangifera indica",
        "family": "Anacardiaceae",
        "description": (
            "The 'King of Fruits', native to South Asia, prized for its juicy "
            "sweet fruit and wide range of therapeutic uses across all plant parts."
        ),
        "benefits": [
            "High in Vitamins A, C, and E",
            "Rich in dietary fibre",
            "Contains mangiferin — a potent antioxidant",
            "Boosts immunity",
            "Supports eye health",
        ],
        "medicinal_uses": [
            "Mango leaf tea used for diabetes and hypertension",
            "Bark extract used for toothache and gum problems",
            "Kernel used in Ayurveda for diarrhoea and leucorrhoea",
        ],
        "agricultural_uses": [
            "Deep roots improve soil aeration",
            "Fallen leaves decompose into nitrogen-rich compost",
            "Shade tree for intercropping",
        ],
        "vrikshayurveda": (
            "Considered the 'Kalpataru' (wish-fulfilling tree) in ancient texts; "
            "mango wood, leaves, and fruit hold prominent roles in Vedic rituals."
        ),
    },

    "Neem": {
        "common_name": "Neem",
        "scientific_name": "Azadirachta indica",
        "family": "Meliaceae",
        "description": (
            "One of the most versatile medicinal trees in Ayurveda, revered as "
            "'the village pharmacy' for its multi-purpose therapeutic properties."
        ),
        "benefits": [
            "Powerful antibacterial and antiviral agent",
            "Natural blood purifier",
            "Supports skin health",
            "Effective natural pesticide",
            "Anti-diabetic properties",
        ],
        "medicinal_uses": [
            "Neem twigs used as toothbrush (datun)",
            "Leaf paste applied for acne and skin infections",
            "Oil used to treat lice and dandruff",
            "Bark decoction for malaria fever",
        ],
        "agricultural_uses": [
            "Neem cake (seed press residue) used as organic fertiliser",
            "Neem oil spray — broad-spectrum biopesticide",
            "Soil amendment to suppress nematodes",
        ],
        "vrikshayurveda": (
            "Vrikshayurveda prescribes Neem leaf slurry as a universal plant "
            "tonic to cure tree diseases; it is called 'Sarva Roga Nivarini' "
            "(cure of all diseases) in ancient texts."
        ),
    },

    "Rose": {
        "common_name": "Rose",
        "scientific_name": "Rosa damascena",
        "family": "Rosaceae",
        "description": (
            "One of the oldest cultivated flowers, celebrated for its fragrance, "
            "beauty, and extensive use in cosmetics and traditional medicine."
        ),
        "benefits": [
            "Rich in Vitamin C and antioxidants",
            "Anti-inflammatory and analgesic properties",
            "Calming effect on the nervous system",
            "Promotes skin hydration",
            "Antibacterial activity",
        ],
        "medicinal_uses": [
            "Rose water used as eye wash and skin toner",
            "Rose hip tea for immunity and joint pain",
            "Petals used in gulkand for acidity and constipation",
        ],
        "agricultural_uses": [
            "Companion planting to attract pollinators",
            "Rose compost improves soil microbial diversity",
        ],
        "vrikshayurveda": (
            "Described in Dhanvantari Nighantu, rose (Shatapatra) is used for "
            "heart ailments and mental disorders in classical Ayurvedic formulations."
        ),
    },

    "Tulsi": {
        "common_name": "Holy Basil (Tulsi)",
        "scientific_name": "Ocimum tenuiflorum",
        "family": "Lamiaceae",
        "description": (
            "Considered the most sacred plant in Hinduism; an adaptogenic herb "
            "with remarkable medicinal properties recognised by modern science."
        ),
        "benefits": [
            "Powerful adaptogen (reduces stress and anxiety)",
            "Antimicrobial, antiviral, and antibacterial",
            "Boosts immunity",
            "Regulates blood sugar levels",
            "Anti-inflammatory and analgesic",
        ],
        "medicinal_uses": [
            "Kadha (decoction) for respiratory infections and cold",
            "Leaf juice for fever, indigestion, and skin diseases",
            "Ear drops from leaf extract for ear infections",
            "Seed (Sabja) for cooling and digestion",
        ],
        "agricultural_uses": [
            "Natural insect and mosquito repellent when planted around crops",
            "Improves soil microbial activity",
        ],
        "vrikshayurveda": (
            "Tulsi occupies the highest position in Vrikshayurveda; planting it "
            "near grain stores is recommended to prevent pests and preserve seeds. "
            "Every home was prescribed a Tulsi plant as a living pharmacy."
        ),
    },

    "Turmeric": {
        "common_name": "Turmeric",
        "scientific_name": "Curcuma longa",
        "family": "Zingiberaceae",
        "description": (
            "A golden-yellow rhizomatous plant widely used as a spice, dye, and "
            "medicine, with curcumin as its primary bioactive compound."
        ),
        "benefits": [
            "Potent anti-inflammatory (curcumin)",
            "Strong antioxidant properties",
            "Supports liver detoxification",
            "Boosts brain-derived neurotrophic factor (BDNF)",
            "Aids wound healing",
        ],
        "medicinal_uses": [
            "Golden milk (turmeric latte) for immunity and inflammation",
            "Paste applied on wounds and skin infections",
            "Used in Ayurveda for jaundice, anaemia, and arthritis",
        ],
        "agricultural_uses": [
            "Turmeric powder used as natural fungicide on crops",
            "Rhizome juice deters soil pests",
        ],
        "vrikshayurveda": (
            "Haridra (Turmeric) is mentioned in Vrikshayurveda as a plant "
            "protectant; its paste mixed with cow dung is applied to tree wounds."
        ),
    },

    "Almond": {
        "common_name": "Almond",
        "scientific_name": "Prunus dulcis",
        "family": "Rosaceae",
        "description": (
            "A nutrient-dense tree nut and the world's most widely grown and "
            "consumed tree nut, rich in healthy fats, protein, and micronutrients."
        ),
        "benefits": [
            "High in Vitamin E and magnesium",
            "Promotes heart health by reducing LDL cholesterol",
            "Supports brain function and memory",
            "Regulates blood sugar",
            "Rich source of plant-based protein",
        ],
        "medicinal_uses": [
            "Almond oil used for dry skin and hair care",
            "Sweet almond paste for respiratory conditions in Unani medicine",
            "Used in Ayurveda for vata pacification",
        ],
        "agricultural_uses": [
            "Almond shells used as mulch and biomass fuel",
            "Honey bee forage — significant pollinator support crop",
        ],
        "vrikshayurveda": (
            "Vatama (Almond) is listed in Charaka Samhita as a brain tonic; "
            "soaked almonds with milk are prescribed for memory enhancement."
        ),
    },

    "Papaya": {
        "common_name": "Papaya",
        "scientific_name": "Carica papaya",
        "family": "Caricaceae",
        "description": (
            "A fast-growing tropical fruit tree famed for its orange-yellow flesh "
            "and powerful digestive enzyme papain."
        ),
        "benefits": [
            "Rich in Vitamin C and beta-carotene",
            "Contains papain — a digestive enzyme",
            "Anti-malarial properties (leaf extract)",
            "Supports platelet production",
            "Anti-cancer potential (antioxidants)",
        ],
        "medicinal_uses": [
            "Papaya leaf juice used to increase platelet count in dengue",
            "Papain supplements for digestive disorders",
            "Unripe papaya used traditionally as abortifacient",
        ],
        "agricultural_uses": [
            "Papain extracted commercially as meat tenderiser",
            "Leaves used as natural fertiliser (high N content)",
        ],
        "vrikshayurveda": (
            "Though introduced post-Columbus, papaya rapidly integrated into "
            "Ayurvedic practice; it is now used in liver tonic syrups and "
            "digestive preparations across Indian traditional medicine."
        ),
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Resize and normalise a PIL Image for MobileNetV2 input.

    Args:
        image: A PIL.Image object (any mode).

    Returns:
        np.ndarray of shape (1, 224, 224, 3), values in [0, 1].
    """
    # Convert to RGB to ensure 3-channel input (handles RGBA / grayscale)
    image = image.convert("RGB")
    # Resize to model's expected input dimensions
    image = image.resize(IMAGE_SIZE)
    # Convert to numpy array and normalise to [0, 1]
    img_array = np.array(image, dtype=np.float32) / 255.0
    # Add batch dimension → (1, 224, 224, 3)
    return np.expand_dims(img_array, axis=0)


def get_plant_info(class_name: str) -> dict:
    """
    Retrieve plant information from the database for a given class name.

    Args:
        class_name: String key matching a key in PLANT_INFO.

    Returns:
        Dictionary with plant details, or a default 'unknown' dict if not found.
    """
    return PLANT_INFO.get(
        class_name,
        {
            "common_name": class_name.replace("_", " "),
            "scientific_name": "Unknown",
            "family": "Unknown",
            "description": "Detailed information for this plant is not yet in our database.",
            "benefits": ["Information not available"],
            "medicinal_uses": ["Information not available"],
            "agricultural_uses": ["Information not available"],
            "vrikshayurveda": "Information not available",
        },
    )


def predict_plant(model, image: Image.Image, class_names: list = None):
    """
    Run inference on a PIL Image and return the top predicted class and confidence.

    Args:
        model:       Loaded Keras model.
        image:       PIL.Image object to classify.
        class_names: List of class names (ordered). Falls back to CLASS_NAMES if None.

    Returns:
        Tuple (class_name: str, confidence: float)
        e.g. ("Neem", 0.9432)
    """
    if class_names is None:
        class_names = CLASS_NAMES

    processed = preprocess_image(image)
    predictions = model.predict(processed, verbose=0)  # shape (1, num_classes)
    predicted_idx = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][predicted_idx])
    class_name = class_names[predicted_idx]
    return class_name, confidence


def export_plant_info_json(filepath: str = "plant_database.json") -> None:
    """
    Export the PLANT_INFO dictionary to a JSON file for external use.

    Args:
        filepath: Path to the output JSON file.
    """
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(PLANT_INFO, f, indent=4, ensure_ascii=False)
    print(f"Plant database exported to: {filepath}")
