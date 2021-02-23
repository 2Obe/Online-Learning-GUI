
#%% Import libraries:
from keras.layers import (Reshape, BatchNormalization, Conv2D, LeakyReLU,
                          MaxPool2D, Flatten, Dense, Conv2DTranspose,
                          Activation, Dropout, ActivityRegularization)
from keras import models
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import keras

# %% Define Class for Models:
class EmbedderModeler():
    '''
    Class for a keras Embeddingmodel bale to train an AE and Classifier
    '''

    def __init__(
            self, train_path, val_path=None, nr_dense=2, batch_size=32,
            size_dense=32, dropout=0.5, dim=200, epochs=20, patience=4):
        self.train_path = train_path
        self.val_path = val_path
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.nr_dense = nr_dense
        self.size_dense = size_dense
        self.dropout = dropout
        self.n = dim

    def generator(self, val_split=0.2, horizontal_flip=False, rotation=180, rescale=1./1.):
        '''
        Creates new Datagenerator normalizing splitting input images
        '''
        # Define datagenerator:
        datagen = ImageDataGenerator(rotation_range=rotation,
                                     horizontal_flip=horizontal_flip,
                                     # vertical_flip=True,
                                     rescale=rescale,
                                     validation_split=val_split,
                                     )
        #datagen = ImageDataGenerator(validation_split=val_split)

        return datagen

    def callbacks(self, mc_monitor='val_loss', NAME="",):
        '''
        Defines list of callbacks, 
        NAME ("best_model_AE" or "best_model_Classifier" )
        is name of best model stored as Model checkpoint
        mc_monitor means metric to monitor to save best model
        '''
        # Define Callbacks:
        # Early stopping:
        ES_callback = keras.callbacks.EarlyStopping(monitor=mc_monitor,
                                                    patience=self.patience)
        # Learning rate scheduler:
        LR_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                           factor=0.1,
                                                           patience=2,
                                                           verbose=1,
                                                           epsilon=1e-4,
                                                           mode='min')
        # Tensorboard:
        TB_callback = TensorBoard(log_dir='logs')
        # Model Checkpoint Callback:
        MC_callback = tf.keras.callbacks.ModelCheckpoint(
            (NAME),
            save_weights_only=False,
            monitor=mc_monitor,
            mode='auto',
            save_best_only=True)

        # Define Callbacklist:
        callback_list = [TB_callback, ES_callback, MC_callback, LR_callback]

        return callback_list

    def train_on_pretrained(self, datagen, pt_model_path, callback_list, pt_model=None):
        '''
        Supervised finetuning of pretrained encoder model (pt_model)
        with added dense classifier.
        Returns best trained Encoder for Embedding extraction
        '''
        # Load pretrained model from AE:
        if pt_model is None:
            pt_model = load_model(pt_model_path)

        pt_model = tf.keras.Sequential(pt_model.layers[:-10])

        # Finetuning of architecture
        number_dense = [self.nr_dense]
        size_dense = [self.size_dense]
        dropout = [self.dropout]

        for number_dense in number_dense:
            for size_dense in size_dense:
                for d in dropout:
                    NAME = "{}-dense-{}-nodes-{}-dropout".format(number_dense,
                                                                 size_dense, d)
                    print(NAME)

                    # Define Model
                    trained_CNN = models.Sequential()
                    # trained_CNN.add(Reshape((90000,1), input_shape=(300,300)))
                    trained_CNN.add(pt_model)
                    trained_CNN.add(Dropout(d))
                    for s_d in range(number_dense):
                        trained_CNN.add(Dense(size_dense, activation='relu'))

                    trained_CNN.add(Dense(1, activation='sigmoid'))

                    # print("model built!")
                    # trained_CNN.summary()
                    # Compile pretrained CNN model
                    trained_CNN.compile(optimizer="adam", loss='binary_crossentropy',
                                        metrics="acc")

                    # [metrics.Precision(), metrics.Recall(),metrics.AUC()])
                    # Define train_generator:
                    # Define train dir:
                    train_generator = datagen.flow_from_directory(self.train_path,
                                                                  target_size=(
                                                                      300, 300),
                                                                  color_mode="grayscale",
                                                                  batch_size=self.batch_size,
                                                                  shuffle=False,
                                                                  class_mode='binary',
                                                                  seed=42,
                                                                  subset='training')
                    # define val_generator:
                    validation_generator = datagen.flow_from_directory(self.train_path,
                                                                       target_size=(
                                                                           300, 300),
                                                                       color_mode="grayscale",
                                                                       batch_size=self.batch_size,
                                                                       shuffle=False,
                                                                       class_mode='binary',
                                                                       seed=42,
                                                                       subset='validation')

                    # reset generators:
                    train_generator.reset()
                    validation_generator.reset()

                    # Train pretrained CNN model
                    history = trained_CNN.fit(
                        train_generator,
                        steps_per_epoch=int(
                            train_generator.samples/self.batch_size),
                        validation_data=validation_generator,
                        validation_steps=int(
                            validation_generator.samples/self.batch_size),
                        epochs=self.epochs,
                        callbacks=callback_list
                    )
                    # history = trained_CNN.fit_generator(train_generator,
                    # steps_per_epoch=1985,
                    # epochs=self.epochs)

                    # Slice all dense layers for obtaining only trained encoder base:
                    trained_encoder = keras.models.load_model(
                        '/content/best_model_Classifier')
                    #trained_encoder = tf.keras.Sequential(best_model.layers[:-4])
                    # trained_encoder.summary()

                    return trained_encoder

        # def predicting(trained_model):

    def train_AE(self, datagen, callback_list):
        '''
        Pretrains an CNN AE model and returns pretrained encoder part
        INPUT: datagenerator from train generator
        OUTPUT: best pretrained Encoder model of CNN AE
        '''
        # Define Model:
        inp = tf.keras.Input(shape=(300, 300), name="inputs")
        x_reshape = Reshape((300, 300, 1))(inp)
        x2 = Conv2D(128, (3, 3), padding="same")(x_reshape)
        x3 = BatchNormalization()(x2)
        x4 = LeakyReLU(alpha=0.2)(x3)
        x5 = MaxPool2D((2, 2))(x4)
        #x2n = Conv2D(128, (3, 3), padding="same")(x5)
        #x3n = BatchNormalization()(x2n)
        #x4n = LeakyReLU(alpha=0.2)(x3n)

        x6 = Conv2D(128, (3, 3), padding="same")(x5)
        x7 = BatchNormalization()(x6)
        x8 = LeakyReLU(alpha=0.2)(x7)
        x9 = MaxPool2D((2, 2))(x8)
        # Latent space encoded as Dense units (n-dimensional)
        x10 = Flatten()(x9)
        xreg = ActivityRegularization(l1=1e-3)(x10)
        new_shape = int(300/4)
        units = int(new_shape * new_shape * 64)
        x11 = Dense(self.n, name="latent")(xreg)
        x12 = Dense(units)(x11)
        x13 = LeakyReLU(alpha=0.2)(x12)
        x14 = Reshape((new_shape, new_shape, 64))(x13)

        # Decoder
        x15 = Conv2DTranspose(128, (3, 3), strides=2, padding="same")(x14)
        x16 = BatchNormalization()(x15)
        x17 = LeakyReLU(alpha=0.2)(x16)

        x18 = Conv2DTranspose(1, (3, 3), strides=2, padding="same")(x17)
        x19 = BatchNormalization()(x18)
        x20 = Activation("sigmoid", name="outputs")(x19)

        outputs = Reshape((300, 300))(x20)

        # Define CNN AE model
        CNN_AE = tf.keras.Model(inp, outputs)
        CNN_AE.summary()

        # Compile CNN AE:
        CNN_AE.compile(optimizer="adam", loss='binary_crossentropy')

        # Define train dir:
        train_generator = datagen.flow_from_directory(self.train_path,
                                                      target_size=(300, 300),
                                                      color_mode="grayscale",
                                                      batch_size=self.batch_size,
                                                      class_mode='input',
                                                      seed=42,
                                                      subset='training')
        # define val_generator:
        validation_generator = datagen.flow_from_directory(self.train_path,
                                                           target_size=(
                                                               300, 300),
                                                           color_mode="grayscale",
                                                           batch_size=self.batch_size,
                                                           class_mode='input',
                                                           seed=42,
                                                           subset='validation')
        # reset generators:
        train_generator.reset()
        validation_generator.reset()
        # Train CNN AE model
        history = CNN_AE.fit(
            train_generator,
            steps_per_epoch=int(train_generator.samples/self.batch_size),
            validation_data=validation_generator,
            validation_steps=int(validation_generator.samples/self.batch_size),
            epochs=self.epochs,
            callbacks=callback_list
        )

        best_model = keras.models.load_model('/content/best_model_AE')
        pretrained_base = tf.keras.Sequential(best_model.layers[:-10])
        # pretrained_base.summary()

        return pretrained_base
