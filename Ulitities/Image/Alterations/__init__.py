from Ulitities.Image.Alterations.GaussianNoiseAlteration import GaussianNoiseAlteration
from Ulitities.Image.Alterations.JpegCompressionAlteration import JpegCompressionAlteration

available_alterations = [GaussianNoiseAlteration, JpegCompressionAlteration]

def choose_alteration():
    text = "Choose the next alteration to apply: \n 0) None"
    i = 1
    for alteration in available_alterations:
        text += "\n {}) {}".format(i, alteration.name)
        i += 1

    i = int(input(text+"\n"))

    if i <= 0 :
        return None
    else:
        return available_alterations[i-1](requires_inputs=True)
