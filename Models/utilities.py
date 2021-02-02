from Datasets.CASIA2 import CASIA2
from Geneartors.CASIA2.Casia2Generator import Casia2Generator


def test_model(model,input_type,tests,batch_size = 32):
    dataset = CASIA2()
    dataset.download_and_prepare()

    result = [i for i in range(len(tests))]

    for idx in range(len(tests)):
        generator_test = Casia2Generator(dataset.as_dataset(split=tests[idx]),input_type, batch_size)
        result[idx] = model.evaluate(generator_test)[1]

    return result