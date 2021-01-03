import time
from abc import ABC, abstractmethod
from pathlib import Path

from tensorflow.python.keras.models import Sequential
import tensorflow as tf
from generators import DataGenerator


class BaseModel(ABC):
    """
        Base class defining the endpoint to use to interact with a model
    """
    def __init__(self,model_name:str,log_dir:Path,input_shape,output_shape,verbose:bool=True):
        """
        :param model_name: name of the model, used for logging and saving it
        :param log_dir: path of the dir in which to save the model and the tensorboard log
        :param verbose: boolean indicating if it is necessary to print extensive information
            in the console
        """

        #check if the log folder is a valid folder
        assert(log_dir.is_dir())

        self.name = model_name
        self.parent_log_dir = log_dir
        self.verbose = verbose

        #generating a folder in which to save the data of this model
        str_time = time.strftime("%b %d %Y %H:%M:%S", time.gmtime())
        self.log_dir = self.parent_log_dir / self.name / str_time

        if not self.log_dir.is_dir():
            self.log_dir.mkdir(parents=True, exist_ok=True)

        #generating a unique name for the model depending on the time of its creation
        self.name_with_time = self.name +" "+ str_time

        self.input_shape = input_shape
        self.output_shape = output_shape


    @abstractmethod
    def build_model(self,input_shape,output_shape) -> Sequential:
        """
        Function in charge of defining the model structure
        :param input_shape: tuple containing the shape of the data this model will recive as input
        :param output_shape: tuple containing the shape of the output produced by this model
        :return: Keras Sequential Model
        """
        raise NotImplementedError

    def _get_callbacks(self) -> list:
        """
        Function defining all the callbacks for the given model and returning them as a list.
        In particular by default each model uses the following 3 callbacks
            - early stopping -> to stop the train early if the model has not improved in the past 10 epochs
            - checkpoint -> to save the model each time we find better weights
            - tensorboard -> to save the model logs and be able to confront the models
        :return: list(keras.Callbacks)
        """
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=10),
            tf.keras.callbacks.ModelCheckpoint(filepath=self.log_dir / "checkpoints"/'model{epoch:02d}-{val_loss:.2f}.h5'),
            tf.keras.callbacks.TensorBoard(log_dir=self.parent_log_dir / "tensorboard"),
        ]
        return  callbacks

    def _on_before_train(self):
        """
        Set of actions to do right before the training phase
        :return:
        """
        self.training_start_time = time.time()

        if self.verbose:
            print("The training phase of the model {} has started at:{}".format(self.name,self.training_start_time))

    def _on_after_train(self):
        """
        Set of actions to do right after the training phase
        :return:
        """
        self.training_time = time.time() - self.training_start_time

        if self.verbose:
            print("The model:{} has completed the training phase in: {}".format(self.name,self.training_time))

    def train_model(self,training_data : DataGenerator,validation_data : DataGenerator, epochs:int,
                    optimizer = tf.keras.optimizers.Adam(lr=0.0001),
                    loss_function = 'categorical_crossentropy',
                    save:bool = False):
        """
        Function in charge of training the model defined in the given class
        :param training_data: DataGenerator class, generating the training data
        :param validation_data: Datagenerator class, generating the validation data
        :param optimizer: optimizer to use during training,
        :param loss_function: loss function to use
        :param epochs: number of epochs to run
        :param save: should the model be saved at the end of the training phase?
        :return:
        """

        #get the structure of the model as defined by the build function
        self.model = self.build_model(self.input_shape,self.output_shape)

        #compile the model
        self.model.compile(optimizer=optimizer,loss=loss_function,metrics=['accuracy'])

        #execute "on before train" operations
        self._on_before_train()

        #execute "on after train" operations
        self.model.fit_generator(training_data,steps_per_epoch=len(training_data), epochs=epochs,
                                 validation_data=validation_data, validation_steps=len(validation_data),
                                 callbacks=self._get_callbacks())

        #execute "on after train" operations
        self._on_after_train()

        #save the final model
        if save:
            self.model.save(self.log_dir / "final-model.{val_loss:.2f}.h5")