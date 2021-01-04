import os
import time
from abc import ABC, abstractmethod
from contextlib import redirect_stdout
from pathlib import Path

from tensorflow.python.keras.models import Sequential
import tensorflow as tf
from Geneartors import DataGenerator


class BaseModel(ABC):
    """
        Base class defining the endpoint to use to interact with a model
    """

    def __init__(self, model_name: str, log_dir: Path, input_shape, output_shape, verbose: bool = True):
        """
        :param model_name: name of the model, used for logging and saving it
        :param log_dir: path of the dir in which to save the model and the tensorboard log
        :param verbose: boolean indicating if it is necessary to print extensive information
            in the console
        """

        # check if the log folder is a valid folder
        assert (log_dir.is_dir())

        # the verbose parameter controls how many log info will be printed in the console
        self.verbose = verbose

        # save the time of creation of this class, it will help us to uniquelly identify this specific train run
        self.str_time = time.strftime("%b %d %Y %H:%M:%S", time.gmtime())

        # save the model name and the directory in which to save the Logs
        self.name = model_name
        self.parent_log_dir = log_dir

        # create the path of the log folder for this train run
        self.log_dir = self.parent_log_dir / "models" / self.name / self.str_time

        # create the log folder
        if not self.log_dir.is_dir():
            self.log_dir.mkdir(parents=True, exist_ok=True)

        # tensorboard has its own log directory
        self.tensorboard_log_dir = self.parent_log_dir / "tensorboard" / self.name / self.str_time

        # generating a unique name for the model depending on the time of its creation
        self.name_with_time = self.name + " " + self.str_time

        self.input_shape = input_shape
        self.output_shape = output_shape

    @abstractmethod
    def build_model(self, input_shape, output_shape) -> Sequential:
        """
        Function in charge of defining the model structure
        :param input_shape: tuple containing the shape of the data this model will recive as input
        :param output_shape: tuple containing the shape of the output produced by this model
        :return: Keras Sequential Model
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def input_shape(self) -> tuple:
        """
        This property returns the input shape of the model
        :return: tuple
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def output_shape(self) -> tuple:
        """
        This property returns the output shape of the model
        :return:
        """
        raise NotImplementedError

    def _get_callbacks(self) -> list:
        """
        Function defining all the callbacks for the given model and returning them as a list.
        In particular by default each model uses the following 3 callbacks
            - early stopping -> to stop the train early if the model has not improved in the past 10 epochs
            - checkpoint -> to save the model each time we find better weights
            - tensorboard -> to save the model Logs and be able to confront the models
        :return: list(keras.Callbacks)
        """
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=10),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=self.log_dir / "checkpoints" / 'model{epoch:02d}-{val_loss:.2f}.h5'),
            tf.keras.callbacks.TensorBoard(log_dir=self.tensorboard_log_dir),
        ]
        return callbacks

    def _on_before_train(self):
        """
        Set of actions to do right before the training phase
        :return:
        """
        self.training_start_time = time.time()

        if self.verbose:
            print("The training phase of the model {} has started at:{}".format(self.name, self.training_start_time))

    def _on_after_train(self):
        """
        Set of actions to do right after the training phase
        :return:
        """
        self.training_time = time.time() - self.training_start_time

        if self.verbose:
            print("The model:{} has completed the training phase in: {}".format(self.name, self.training_time))

    def train_model(self, training_data: DataGenerator, validation_data: DataGenerator, epochs: int, loss_function,
                    optimizer=tf.keras.optimizers.Adam(lr=0.0001),
                    save_model: bool = False, save_summary: bool = True):
        """
        Function in charge of training the model defined in the given class
        :param training_data: DataGenerator class, generating the training data
        :param validation_data: Datagenerator class, generating the validation data
        :param optimizer: optimizer to use during training,
        :param loss_function: loss function to use
        :param epochs: number of epochs to run
        :param save_model: should the model be saved at the end of the training phase?
        :param save_summary: save the summary of the model into the log folder
        :return:
        """

        # get the structure of the model as defined by the build function
        self.model = self.build_model(self.input_shape, self.output_shape)

        # compile the model
        self.model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])

        # save the summary of the model if required
        if save_summary:
            with open(self.log_dir / 'summary.txt', 'w') as f:
                with redirect_stdout(f):
                    self.model.summary()

        # execute "on before train" operations
        self._on_before_train()

        # train the model
        self.model.fit(training_data, steps_per_epoch=len(training_data), epochs=epochs,
                       validation_data=validation_data, validation_steps=len(validation_data),
                       callbacks=self._get_callbacks(), workers=4)

        # execute "on after train" operations
        self._on_after_train()

        # save the final model
        if save_model:
            self.model.save(self.log_dir / "final-model.{val_loss:.2f}.h5")

    def __del__(self, exc_type, exc_val, exc_tb):
        """
        On deleting the instance of this model, check if its log folder is empty, if it is, delete it to keep
        the Logs as clean as possible
        :return:
        """

        # check if the folder exists
        if not self.log_dir.is_dir():
            return

        # get the content of the folder as a list of paths
        dirContents = os.listdir(self.log_dir)

        # if the content list is empty delete the folder
        if len(dirContents) == 0:
            os.rmdir(self.log_dir)
