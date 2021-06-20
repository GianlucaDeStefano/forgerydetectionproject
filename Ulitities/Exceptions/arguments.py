class InvalidArgumentException(Exception):
    def __init__(self, reason):
        super().__init__("Invalid argument combination, provide: \n"
                         "   1) -i <Name of the image> \n"
                         "   2) -i <Path to the image> -g < path to Binary mask of the image (1->forged area, 0->authentic "
                         "area)>")
