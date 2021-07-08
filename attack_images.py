from Attacks import supported_attacks
from attack_image import attack_image

#images to try to attack
images = []

#select an attack to perform on all the images
print("\nSelect the attack to perform:")
for key, supported_attack in supported_attacks.items():
    print("  {}) {}".format(i, key))
    i = i + 1
attack_type = supported_attacks.keys[int(input("Enter attack number:"))]

for image in images:
    attack_image(image,attack_type=attack_type)