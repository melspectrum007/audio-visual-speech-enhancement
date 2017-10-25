import argparse
import os
from datetime import datetime
import logging

import numpy as np
import matplotlib.pyplot as plt
import data_processor
from dataset import AudioVisualDataset, AudioDataset
from network import SpeechEnhancementNetwork

from mediaio import ffmpeg
from mediaio.video_io import VideoFileWriter



def preprocess(args):
    speaker_ids = list_speakers(args)

    video_file_paths, speech_file_paths, noise_file_paths = list_data(
        args.dataset_dir, speaker_ids, args.noise_dirs, max_files=1500
    )

    video_samples, mixed_spectrograms, speech_spectrograms, noise_spectrograms = data_processor.preprocess_data(
        video_file_paths, speech_file_paths, noise_file_paths
    )

    np.savez(
        args.preprocessed_blob_path,
        video_samples=video_samples,
        mixed_spectrograms=mixed_spectrograms,
        speech_spectrograms=speech_spectrograms,
        noise_spectrograms=noise_spectrograms
    )


def load_preprocessed_samples(preprocessed_blob_path, max_samples=None):
    with np.load(preprocessed_blob_path) as data:
        video_samples = data["video_samples"]
        mixed_spectrograms = data["mixed_spectrograms"]
        speech_spectrograms = data["speech_spectrograms"]
        noise_spectrograms = data["noise_spectrograms"]

    permutation = np.random.permutation(video_samples.shape[0])
    video_samples = video_samples[permutation]
    mixed_spectrograms = mixed_spectrograms[permutation]
    speech_spectrograms = speech_spectrograms[permutation]
    noise_spectrograms = noise_spectrograms[permutation]

    return (
        video_samples[:max_samples],
        mixed_spectrograms[:max_samples],
        speech_spectrograms[:max_samples],
        noise_spectrograms[:max_samples]
    )


def train(args):
    video_samples, mixed_spectrograms, speech_spectrograms, _ = load_preprocessed_samples(args.preprocessed_blob_path)

    normalized_video_samples = np.copy(video_samples)

    normalization_data = data_processor.DataNormalizer.normalize(normalized_video_samples, mixed_spectrograms)
    normalization_data.save(args.normalization_cache)

    # replace some mixed with clean spectrograms
    n_train = mixed_spectrograms.shape[0]
    n_clean = int(args.clean_split * n_train)
    ind = np.random.choice(n_train, n_clean, replace=False)
    mixed_spectrograms[ind] = speech_spectrograms[ind]

    # network = SpeechEnhancementNetwork.build(mixed_spectrograms.shape[1:], video_samples.shape[1:])
    network = SpeechEnhancementNetwork.load(args.model_cache_dir)
    network.train(
        mixed_spectrograms, normalized_video_samples,
        speech_spectrograms, video_samples,
        args.model_cache_dir, args.tensorboard_dir
    )

    network.save(args.model_cache_dir)


def predict(args):
    storage = PredictionStorage(args.prediction_output_dir)

    network = SpeechEnhancementNetwork.load(args.model_cache_dir)
    normalization_data = data_processor.NormalizationData.load(args.normalization_cache)

    speaker_ids = list_speakers(args)
    for speaker_id in speaker_ids:
        video_file_paths, speech_file_paths, noise_file_paths = list_data(
            args.dataset_dir, [speaker_id], args.noise_dirs, max_files=10
        )

        for video_file_path, speech_file_path, noise_file_path in zip(video_file_paths, speech_file_paths, noise_file_paths):
            try:
                print("predicting (%s, %s)..." % (video_file_path, noise_file_path))

                video_samples, mixed_spectrograms, _, _, mixed_signal, video_frame_rate = data_processor.preprocess_sample(
                    video_file_path, speech_file_path, noise_file_path
                )

                data_processor.DataNormalizer.apply_normalization(video_samples, mixed_spectrograms, normalization_data)

                predicted_speech_spectrograms, recovered_video_samples = network.predict(mixed_spectrograms, video_samples)
                # print predicted_speech_spectrograms.mean(), predicted_speech_spectrograms.max(), predicted_speech_spectrograms.min()
                # N, f, t, = predicted_speech_spectrograms.shape


                predicted_speech_signal = data_processor.reconstruct_speech_signal(
                    mixed_signal, predicted_speech_spectrograms, video_frame_rate
                )

                storage.save_prediction(
                    speaker_id, video_file_path, noise_file_path,
                    mixed_signal, predicted_speech_signal,
                    video_frame_rate, recovered_video_samples
                )

            except Exception:
                logging.exception("failed to predict %s. skipping" % video_file_path)


class PredictionStorage(object):

    def __init__(self, storage_dir):
        self.__base_dir = os.path.join(storage_dir, '{:%Y-%m-%d_%H-%M-%S}'.format(datetime.now()))
        os.mkdir(self.__base_dir)

    def __create_speaker_dir(self, speaker_id):
        speaker_dir = os.path.join(self.__base_dir, speaker_id)

        if not os.path.exists(speaker_dir):
            os.mkdir(speaker_dir)

        return speaker_dir

    def save_prediction(self, speaker_id, video_file_path, noise_file_path,
                        mixed_signal, predicted_speech_signal,
                        video_frame_rate, recovered_video_samples):

        speaker_dir = self.__create_speaker_dir(speaker_id)

        speech_name = os.path.splitext(os.path.basename(video_file_path))[0]
        noise_name = os.path.splitext(os.path.basename(noise_file_path))[0]

        sample_prediction_dir = os.path.join(speaker_dir, speech_name + "_" + noise_name)
        os.mkdir(sample_prediction_dir)

        mixture_audio_path = os.path.join(sample_prediction_dir, "mixture.wav")
        enhanced_speech_audio_path = os.path.join(sample_prediction_dir, "enhanced.wav")

        mixed_signal.save_to_wav_file(mixture_audio_path)
        predicted_speech_signal.save_to_wav_file(enhanced_speech_audio_path)

        video_extension = os.path.splitext(os.path.basename(video_file_path))[1]
        mixture_video_path = os.path.join(sample_prediction_dir, "mixture" + video_extension)
        enhanced_speech_video_path = os.path.join(sample_prediction_dir, "enhanced" + video_extension)
        temp_recovered_video_path = os.path.join(sample_prediction_dir, "tmp_recovered" + video_extension)
        recovered_video_path = os.path.join(sample_prediction_dir, "recovered" + video_extension)

        with VideoFileWriter(temp_recovered_video_path, video_frame_rate) as writer:
            for s in range(recovered_video_samples.shape[0]):
                for f in range(recovered_video_samples.shape[3]):
                    writer.write_frame(recovered_video_samples[s, :, :, f])

        ffmpeg.merge(video_file_path, mixture_audio_path, mixture_video_path)
        ffmpeg.merge(video_file_path, enhanced_speech_audio_path, enhanced_speech_video_path)
        ffmpeg.merge(temp_recovered_video_path, enhanced_speech_audio_path, recovered_video_path)

        os.unlink(mixture_audio_path)
        # os.unlink(enhanced_speech_audio_path)
        os.unlink(temp_recovered_video_path)


def list_speakers(args):
    if args.speakers is None:
        dataset = AudioVisualDataset(args.dataset_dir)
        speaker_ids = dataset.list_speakers()
    else:
        speaker_ids = args.speakers

    if args.ignored_speakers is not None:
        for speaker_id in args.ignored_speakers:
            speaker_ids.remove(speaker_id)

    return speaker_ids


def list_data(dataset_dir, speaker_ids, noise_dirs, max_files=None, self_noise_split=0.5):
    speech_dataset = AudioVisualDataset(dataset_dir)
    speech_subset = speech_dataset.subset(speaker_ids, max_files, shuffle=True)

    noise_dataset = AudioDataset(noise_dirs)
    noise_file_paths = noise_dataset.subset(max_files, shuffle=True)
    n_files = min(speech_subset.size(), len(noise_file_paths))
    if self_noise_split > 0.0:
        self_noise_paths = speech_subset.audio_paths()
        n_self_noise = int(n_files * self_noise_split)
        noise_file_paths[:n_self_noise] = self_noise_paths[-n_self_noise:]


    return speech_subset.video_paths()[:n_files], speech_subset.audio_paths()[:n_files], noise_file_paths[:n_files]


def main():
    parser = argparse.ArgumentParser(add_help=False)
    action_parsers = parser.add_subparsers()

    preprocess_parser = action_parsers.add_parser("preprocess")
    preprocess_parser.add_argument("--dataset_dir", type=str, required=True)
    preprocess_parser.add_argument("--noise_dirs", nargs="+", type=str, required=True)
    preprocess_parser.add_argument("--preprocessed_blob_path", type=str, required=True)
    preprocess_parser.add_argument("--speakers", nargs="+", type=str)
    preprocess_parser.add_argument("--ignored_speakers", nargs="+", type=str)
    preprocess_parser.set_defaults(func=preprocess)

    train_parser = action_parsers.add_parser("train")
    train_parser.add_argument("--preprocessed_blob_path", type=str, required=True)
    train_parser.add_argument("--normalization_cache", type=str, required=True)
    train_parser.add_argument("--model_cache_dir", type=str, required=True)
    train_parser.add_argument("--tensorboard_dir", type=str, required=True)
    train_parser.add_argument('--clean_split', type=float, default=0.0)
    # train_parser.add_argument("--speakers", nargs="+", type=str)
    # train_parser.add_argument("--ignored_speakers", nargs="+", type=str)
    train_parser.set_defaults(func=train)

    predict_parser = action_parsers.add_parser("predict")
    predict_parser.add_argument("--dataset_dir", type=str, required=True)
    predict_parser.add_argument("--noise_dirs", nargs="+", type=str, required=True)
    predict_parser.add_argument("--model_cache_dir", type=str, required=True)
    predict_parser.add_argument("--normalization_cache", type=str, required=True)
    predict_parser.add_argument("--prediction_output_dir", type=str, required=True)
    predict_parser.add_argument("--speakers", nargs="+", type=str)
    predict_parser.add_argument("--ignored_speakers", nargs="+", type=str)
    predict_parser.set_defaults(func=predict)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
