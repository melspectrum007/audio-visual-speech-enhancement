import os

from keras import optimizers, regularizers
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, Deconvolution2D
from keras.layers import Dropout, Flatten, BatchNormalization, LeakyReLU, Reshape, Concatenate
# from keras.layers.merge import concatenate
from keras.models import Model, load_model
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
import keras.backend as K
import numpy as np


class SpeechEnhancementNetwork(object):

    def __init__(self, model, audio_only_model=None, video_only_model=None, audio_embedding_shape=None, video_embedding_shape=None):
        self.__model = model
        self.__audio_only_model = audio_only_model
        self.__video_only_model = video_only_model
        self.__audio_embedding_size = np.prod(audio_embedding_shape)
        self.__video_embedding_size = np.prod(video_embedding_shape)

    @classmethod
    def build(cls, audio_spectrogram_shape, video_shape):
        # append channels axis
        extended_audio_spectrogram_shape = list(audio_spectrogram_shape)
        extended_audio_spectrogram_shape.append(1)

        # create modules of network
        audio_encoder, audio_embedding_shape = cls.__build_audio_encoder(extended_audio_spectrogram_shape)
        video_encoder, video_embedding_shape = cls.__build_video_encoder(video_shape)
        attention, shared_embedding_size = cls.__build_attention(audio_embedding_shape, video_embedding_shape)
        decoder = cls.__build_decoder(shared_embedding_size, audio_embedding_shape, video_embedding_shape, video_shape)

        # create possible inputs
        audio_input = Input(shape=extended_audio_spectrogram_shape)
        video_input = Input(shape=video_shape)
        fake_audio_embedding = Input(shape=(np.prod(audio_embedding_shape),))
        fake_video_embedding = Input(shape=(np.prod(video_embedding_shape),))

        # encoding
        audio_embedding = audio_encoder(audio_input)
        video_embedding = video_encoder(video_input)

        # mixing
        av_embedding = attention(inputs=[audio_embedding, video_embedding])
        a_only_embedding = attention(inputs=[audio_embedding, fake_video_embedding])
        v_only_embedding = attention(inputs=[fake_audio_embedding, video_embedding])

        # decoding
        av_in_a_out, av_in_v_out = decoder(av_embedding)
        a_in_a_out, a_in_v_out = decoder(a_only_embedding)
        v_in_a_out, v_in_v_out = decoder(v_only_embedding)

        # compiling models
        audio_visual_model = Model(inputs=[audio_input, video_input], outputs=[av_in_a_out, av_in_v_out])
        audio_visual_model.compile(loss=['mean_squared_error', 'mean_squared_error'], loss_weights=[1, 0.1], optimizer=optimizers.Adam(lr=5e-4))

        audio_only_model = Model(inputs=[audio_input, fake_video_embedding], outputs=[a_in_a_out, a_in_v_out])
        audio_only_model.compile(loss=['mean_squared_error', 'mean_squared_error'], loss_weights=[1, 0.1], optimizer=optimizers.Adam(lr=5e-4))

        video_only_model = Model(inputs=[fake_audio_embedding, video_input], outputs=[v_in_a_out, v_in_v_out])
        video_only_model.compile(loss=['mean_squared_error', 'mean_squared_error'], loss_weights=[1, 0.1], optimizer=optimizers.Adam(lr=5e-4))

        return SpeechEnhancementNetwork(audio_visual_model, audio_only_model, video_only_model, audio_embedding_shape, video_embedding_shape)

    @classmethod
    def __build_audio_encoder(cls, extended_audio_spectrogram_shape):
        audio_input = Input(shape=extended_audio_spectrogram_shape)

        x = Convolution2D(8, kernel_size=(5, 5), strides=(2, 2), padding='same')(audio_input)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        x = Convolution2D(8, kernel_size=(4, 4), strides=(1, 1), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        x = Convolution2D(16, kernel_size=(4, 4), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        x = Convolution2D(32, kernel_size=(2, 2), strides=(2, 1), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        x = Convolution2D(32, kernel_size=(2, 2), strides=(2, 1), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        x = Convolution2D(64, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)

        audio_embedding_shape = x.shape[1:].as_list()
        audio_embedding = Flatten()(x)

        model = Model(inputs=[audio_input], outputs=[audio_embedding])
        model.summary()

        return model, audio_embedding_shape

    @classmethod
    def __build_video_encoder(cls, video_shape):
        video_input = Input(shape=video_shape)

        x = Convolution2D(128, kernel_size=(5, 5), padding='same')(video_input)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        x = Dropout(0.1)(x)

        x = Convolution2D(128, kernel_size=(5, 5), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        x = Dropout(0.1)(x)

        x = Convolution2D(256, kernel_size=(3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        x = Dropout(0.1)(x)

        x = Convolution2D(256, kernel_size=(3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        x = Dropout(0.1)(x)

        x = Convolution2D(512, kernel_size=(3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        x = Dropout(0.1)(x)

        x = Convolution2D(512, kernel_size=(3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        x = Dropout(0.1)(x)

        x = Convolution2D(512, kernel_size=(3, 3), padding='same')(x)

        video_embedding_shape = x.shape[1:].as_list()

        video_embedding = Flatten()(x)

        model = Model(inputs=[video_input], outputs=[video_embedding])
        model.summary()

        return model, video_embedding_shape

    @classmethod
    def __build_attention(cls, audio_embedding_shape, video_embedding_shape):
        audio_embedding_size = np.prod(audio_embedding_shape)
        video_embedding_size = np.prod(video_embedding_shape)
        shared_embedding_size = (audio_embedding_size + video_embedding_size) / 2

        audio_embedding_input = Input(shape=(audio_embedding_size,))
        video_embedding_input = Input(shape=(video_embedding_size,))

        x = Concatenate()([audio_embedding_input, video_embedding_input])
        x = Dense(shared_embedding_size)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Dropout(0.1)(x)

        x = Dense(shared_embedding_size)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Dropout(0.1)(x)
        shared_embedding_out = Dense(shared_embedding_size)(x)

        model = Model(inputs=[audio_embedding_input, video_embedding_input], outputs=shared_embedding_out)
        model.summary()

        return model, shared_embedding_size

    @classmethod
    def __build_decoder(cls, shared_embedding_size, audio_embedding_shape, video_embedding_shape, video_shape):
        shared_embedding_input = Input(shape=(shared_embedding_size,))

        x = Dense(shared_embedding_size)(shared_embedding_input)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Dropout(0.1)(x)

        x = Dense(shared_embedding_size)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Dropout(0.1)(x)

        audio_embedding_size = np.prod(audio_embedding_shape)
        video_embedding_size = np.prod(video_embedding_shape)

        a = Dense(audio_embedding_size)(x)
        a = Reshape(audio_embedding_shape)(a)
        a = BatchNormalization()(a)
        a = LeakyReLU()(a)
        audio_embedding = Dropout(0.1)(a)

        v = Dense(video_embedding_size)(x)
        v = Reshape(video_embedding_shape)(v)
        v = BatchNormalization()(v)
        v = LeakyReLU()(v)
        video_embedding = Dropout(0.1)(v)

        audio_output = cls.__build_audio_decoder(audio_embedding)
        video_output = cls.__build_video_decoder(video_embedding, video_shape)

        model = Model(inputs=shared_embedding_input, outputs=[audio_output, video_output])
        model.summary()

        return model

    @staticmethod
    def __build_audio_decoder(embedding):
        x = Deconvolution2D(64, kernel_size=(1, 1), strides=(1, 1), padding='same')(embedding)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        x = Deconvolution2D(32, kernel_size=(2, 2), strides=(2, 1), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        x = Deconvolution2D(32, kernel_size=(2, 2), strides=(2, 1), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        x = Deconvolution2D(16, kernel_size=(4, 4), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        x = Deconvolution2D(8, kernel_size=(4, 4), strides=(1, 1), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        x = Deconvolution2D(8, kernel_size=(5, 5), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        x = Deconvolution2D(1, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)

        return x

    @staticmethod
    def __build_video_decoder(embedding, video_shape):
        x = Deconvolution2D(512, kernel_size=(3, 3), padding='same')(embedding)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Dropout(0.1)(x)

        x = Deconvolution2D(512, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Dropout(0.1)(x)

        x = Deconvolution2D(512, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Dropout(0.1)(x)

        x = Deconvolution2D(256, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Dropout(0.1)(x)

        x = Deconvolution2D(256, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Dropout(0.1)(x)

        x = Deconvolution2D(128, kernel_size=(5, 5), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Dropout(0.1)(x)

        x = Deconvolution2D(128, kernel_size=(5, 5), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Dropout(0.1)(x)

        x = Deconvolution2D(video_shape[-1], kernel_size=(1, 1), strides=(1, 1), padding='same')(x)

        return x

    def train(self, mixed_spectrograms, input_video_samples,
              speech_spectrograms, output_video_samples,
              model_cache_dir, tensorboard_dir):

        mixed_spectrograms = np.expand_dims(mixed_spectrograms, -1)  # append channels axis
        speech_spectrograms = np.expand_dims(speech_spectrograms, -1)  # append channels axis

        train_data, val_data = SpeechEnhancementNetwork.__split_train_validation_data([mixed_spectrograms, input_video_samples,
                                                                                       speech_spectrograms, output_video_samples], validation_split=0.1)
        train_mixed_spectrograms, train_input_video_samples, train_speech_spectrograms, train_output_video_samples = train_data
        val_mixed_spectrograms, val_input_video_samples, val_speech_spectrograms, val_output_video_samples = val_data

        model_cache = ModelCache(model_cache_dir)
        audio_video_checkpoint = ModelCheckpoint(model_cache.audio_video_model_path(), verbose=1)
        audio_only_checkpoint = ModelCheckpoint(model_cache.audio_only_model_path(), verbose=1)
        video_only_checkpoint = ModelCheckpoint(model_cache.video_only_model_path(), verbose=1)

        N = mixed_spectrograms.shape[0]

        audio_video_val_loss = np.array([])

        epochs = 400
        epochs_per_mode = 20
        for e in range(epochs / epochs_per_mode):
                print 'audio-video'
                history = self.__model.fit(
                    x=[train_mixed_spectrograms, train_input_video_samples],
                    y=[train_speech_spectrograms, train_output_video_samples],
                    validation_data=[
                        [val_mixed_spectrograms, val_input_video_samples],
                        [val_speech_spectrograms, val_output_video_samples]
                    ],
                    batch_size=16, epochs=epochs_per_mode,
                    callbacks=[audio_video_checkpoint],
                    verbose=1
                )
                audio_video_val_loss = np.append(audio_video_val_loss, history.history['val_loss'])
                if SpeechEnhancementNetwork.check_early_stopping(audio_video_val_loss, patience=epochs_per_mode, delta=0.1):
                    print 'Early stopping'
                    break

                print 'audio-only'
                self.__audio_only_model.fit(
                    x=[train_mixed_spectrograms, np.zeros([N, self.__video_embedding_size])],
                    y=[train_speech_spectrograms, train_output_video_samples],
                    validation_data=[
                        [val_mixed_spectrograms, np.zeros([N, self.__video_embedding_size])],
                        [val_speech_spectrograms, val_output_video_samples]
                    ],
                    batch_size=16, epochs=epochs_per_mode,
                    callbacks=[audio_only_checkpoint],
                    verbose=1
                )

                print 'video-only'
                self.__video_only_model.fit(
                    x=[np.zeros([N, self.__audio_embedding_size]), train_input_video_samples],
                    y=[train_speech_spectrograms, train_output_video_samples],
                    validation_data=[
                        [np.zeros([N, self.__audio_embedding_size]), val_input_video_samples],
                        [val_speech_spectrograms, val_output_video_samples]
                    ],
                    batch_size=16, epochs=epochs_per_mode,
                    callbacks=[video_only_checkpoint],
                    verbose=1
                )

    def predict(self, mixed_spectrograms, video_samples):
        mixed_spectrograms = np.expand_dims(mixed_spectrograms, -1)  # append channels axis
        speech_spectrograms, recovered_video_samples = self.__model.predict([mixed_spectrograms, video_samples])

        return np.squeeze(speech_spectrograms), recovered_video_samples

    @staticmethod
    def load(model_cache_dir):
        model_cache = ModelCache(model_cache_dir)
        auto_encoder = load_model(model_cache.audio_video_model_path())

        return SpeechEnhancementNetwork(auto_encoder)

    def save(self, model_cache_dir):
        model_cache = ModelCache(model_cache_dir)

        self.__model.save(model_cache.audio_video_model_path())

    @staticmethod
    def check_early_stopping(loss, patience, delta):
        head = loss[:-patience]
        tail = loss[-patience:]
        return tail.min() <= head.min() - delta

    @staticmethod
    def __split_train_validation_data(arrays, validation_split):
        n_samples = arrays[0].shape[0]
        permutation = np.random.permutation(n_samples)
        validation_size = int(validation_split * n_samples)
        validation_indices = permutation[:validation_size]
        train_indices = permutation[validation_size:]

        train_arrays = [a[train_indices] for a in arrays]
        validation_arrays = [a[validation_indices] for a in arrays]

        return train_arrays, validation_arrays


class ModelCache(object):

    def __init__(self, cache_dir):
        self.__cache_dir = cache_dir

    def audio_video_model_path(self):
        return os.path.join(self.__cache_dir, "audio_video_model.h5py")

    def audio_only_model_path(self):
        return os.path.join(self.__cache_dir, "audio_only_model.h5py")

    def video_only_model_path(self):
        return os.path.join(self.__cache_dir, "video_only_model.h5py")

