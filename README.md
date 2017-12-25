# Visual Speech Enhancement using Noise-Invariant Training
Implementation of the method described in the paper: [Visual Speech Enhancement using Noise-Invariant Training](http://www.vision.huji.ac.il/speaker-separation) by Aviv Gabbay, Asaph Shamir and Shmuel Peleg.

## Speech Enhancement Demo
<a href="http://www.youtube.com/watch?feature=player_embedded&v=aMrK9PiCHhQ" target="_blank">
<img src="http://img.youtube.com/vi/aMrK9PiCHhQ/0.jpg" width="480" height="360" />
</a>

## Usage
### Dependencies
* python >= 2.7
* [mediaio](https://github.com/avivga/mediaio)
* [face-detection](https://github.com/avivga/face-detection)
* keras >= 2.0.4
* numpy >= 1.12.1
* dlib >= 19.4.0
* opencv >= 3.2.0

### Getting started
Given an audio-visual dataset of the directory structure:
```
├── speaker-1
|   ├── audio
|   |   ├── f1.wav
|   |   └── f2.wav
|   └── video
|	├── f1.mp4
|	└── f2.mp4
├── speaker-2
|   ├── audio
|   |   ├── f1.wav
|   |   └── f2.wav
|   └── video
|	├── f1.mp4
|	└── f2.mp4
...
```
and noise directory contains audio files (*.wav) of noise samples, do the following steps.

Preprocess both train and validation datasets by:
```
speech_enhancer.py preprocess 
	--dataset_dir <path-to-dataset>
	--noise_dirs <path-to-noise-dir> ...
	--preprocessed_blob_path <path-to-preprocessed-output-file>
	[--speakers <speaker-id> ...]
	[--ignored_speakers <speaker-id> ...] 
```

Then, train the model by:
```
speech_enhancer.py train
	--train_preprocessed_blob_paths <paths-to-preprocessed-training-data>
	--validation_preprocessed_blob_paths <paths-to-preprocessed-validation-data>
	--normalization_cache <path-to-save-normalization-data>
	--model_cache_dir <path-to-save-model>
	--tensorboard_dir <path-to-save-tensorboard-stats>
```

Finally, enhance new noisy speech samples by:
```
speech_enhancer.py predict
	--dataset_dir <path-to-dataset>
	--noise_dirs <path-to-noise-dir> ...
	--model_cache_dir <path-to-saved-model>
	--normalization_cache <path-to-saved-normalization-data>
	--prediction_output_dir <path-to-output-predictions>
	[--speakers <speaker-id> ...]
	[--ignored_speakers <speaker-id> ...]
```

## Citing
If you find this project useful for your research, please cite
```
@article{gabbay2017visual,
  title={Visual Speech Enhancement using Noise-Invariant Training},
  author={Gabbay, Aviv and Shamir, Asaph and Peleg, Shmuel},
  journal={arXiv preprint arXiv:1711.08789},
  year={2017}
}
```
