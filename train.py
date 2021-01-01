from datasets.CASIA2 import CASIA2

#import dataset downloading it if necessary
dataset = CASIA2()
dataset.download_and_prepare()