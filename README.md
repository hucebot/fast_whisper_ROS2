# Speech Recognition with Online Whisper

## Project Overview

This project integrates the Faster Whisper ASR (Automatic Speech Recognition) model with ROS 2 to transcribe audio in real-time from a microphone. It leverages faster-whisper, which provides fast and efficient ASR processing, utilizing GPU acceleration where available. The recognized text is then published as a ROS message on a specified topic.

The application is designed for both transcription and optional translation of audio streams, and can be configured for different Whisper model sizes.

## Testing

## Docker

We provide a `Dockerfile` based on `rclpy:iron` which:

- Install the dependencies (CUDA 12, cuDNN 8, faster-whisper)
  - we provide a `CUDNN_ARCH` build argument to select the targeted architecture for cudnn
- Download the faster-whisper large-v3 model from HuggingFace hub
- Use the `whisper_online_mic.py` as entrypoint

###### Build

```
docker build . -t eurobin-whisper
```

default `CUDNN_ARCH` is `linux-x86_64`, you can use `linux-aarch64` for the jetson.

###### Usage

```
docker run --rm --runtime=nvidia --gpus=all --network=host --device=/dev/snd:/dev/snd eurobin-whisper
```

- `--runtime=nvidia` to use the host GPU
- `--device=/dev/snd:/dev/snd` to forward the microphone

You can then pass argument for the `whisper_online_mic.py` as the docker cmd, e.g. `--lang fr` to specify the language used is french.

Watch out, `--lang` is mandatory if you are using the `--translate` option.

One common way of using this node is to ask him to translate to english, hence you can use whatever langage is supported by Whisper, and the resulting message could be plug to any english NLU node (e.g. snips-nlu, llm, and so on).

```
docker run --rm --runtime=nvidia --gpus=all --network=host --device=/dev/snd:/dev/snd eurobin-whisper --translate --lang en
```




### Script

```
usage: whisper_online_mic.py [-h]
                             [--model {tiny.en,tiny,base.en,base,small.en,small,medium.en,medium,large-v1,large-v2,large-v3,large}]
                             [--model_cache_dir MODEL_CACHE_DIR]
                             [--model_dir MODEL_DIR] [--lang LANG]
                             [--buffer_trimming_sec BUFFER_TRIMMING_SEC]
                             [--microphone_blocksize_sec MICROPHONE_BLOCKSIZE_SEC]
                             [--microphone_source MICROPHONE_SOURCE]
                             [-l {DEBUG,INFO,WARN,ERROR,FATAL}]
                             [--ros-topic ROS_TOPIC]
                             [--max-void-count MAX_VOID_COUNT] [--translate]

options:
  -h, --help            show this help message and exit
  --model {tiny.en,tiny,base.en,base,small.en,small,medium.en,medium,large-v1,large-v2,large-v3,large}
                        Name size of the Whisper model to use (default:
                        large-v2). The model is automatically downloaded from
                        the model hub if not present in model cache dir.
  --model_cache_dir MODEL_CACHE_DIR
                        Overriding the default model cache dir where models
                        downloaded from the hub are saved
  --model_dir MODEL_DIR
                        Dir where Whisper model.bin and other files are saved.
                        This option overrides --model and --model_cache_dir
                        parameter.
  --lang LANG, --language LANG
                        Source language code, e.g. en,de,cs, or 'auto' for
                        language detection.
  --buffer_trimming_sec BUFFER_TRIMMING_SEC
                        Buffer trimming length threshold in seconds. If buffer
                        length is longer, trimming sentence/segment is
                        triggered.
  --microphone_blocksize_sec MICROPHONE_BLOCKSIZE_SEC
                        Size of the audio block, default to 10ms.
  --microphone_source MICROPHONE_SOURCE
                        Device name to use.
  -l {DEBUG,INFO,WARN,ERROR,FATAL}, --log-level {DEBUG,INFO,WARN,ERROR,FATAL}
                        Set the log level
  --ros-topic ROS_TOPIC
                        ROS topic on with publish recognized sentences.
  --max-void-count MAX_VOID_COUNT
                        Maximum number of consecutive void recognition before
                        sending the sentence buffer.
  --translate           Explicity activate the translation to the target
                        language (defined by --lang).
```

### Acknowledgements

This work is more or less a ros2 publisher of [whisper_streaming](https://github.com/ufal/whisper_streaming), in case you use this repo please cite [their work](https://aclanthology.org/2023.ijcnlp-demo.3/):
```bibtex
@inproceedings{machacek-etal-2023-turning,
    title = "Turning Whisper into Real-Time Transcription System",
    author = "Mach{\'a}{\v{c}}ek, Dominik  and
      Dabre, Raj  and
      Bojar, Ond{\v{r}}ej",
    editor = "Saha, Sriparna  and
      Sujaini, Herry",
    booktitle = "Proceedings of the 13th International Joint Conference on Natural Language Processing and the 3rd Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics: System Demonstrations",
    month = nov,
    year = "2023",
    address = "Bali, Indonesia",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.ijcnlp-demo.3",
    pages = "17--24",
}
```
