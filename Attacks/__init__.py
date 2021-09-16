from Attacks.BlackBox import black_box_attacks
from Attacks.Exif import exif_attacks
from Attacks.Noiseprint import noiseprint_attacks

families_of_attacks = {
    "Black box": black_box_attacks,
    "Noiseprint based": noiseprint_attacks,
    "Exif-sc based": exif_attacks,
}
