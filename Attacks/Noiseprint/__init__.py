from Attacks.Noiseprint.Lots.Lots4Noiseprint_globalmap import Lots4NoiseprintAttackGlobalMap
from Attacks.Noiseprint.Lots.Lots4Noiseprint_original import Lots4NoiseprintAttackOriginal
from Attacks.Noiseprint.Mimiking.NoiseprintMimickingAttack import NoiseprintMimickingAttack
from Attacks.Noiseprint.Mimiking.NoiseprintMimickingIntelligent import NoiseprintIntelligentMimickingAttack

noiseprint_attacks = {
    "Lots 4 Noiseprint original": Lots4NoiseprintAttackOriginal,
    "Lots 4 Noiseprint globalmap": Lots4NoiseprintAttackGlobalMap,
    "Mimic Noiseprint": NoiseprintMimickingAttack,
    "Mimic Noiseprint Intelligent": NoiseprintIntelligentMimickingAttack
}