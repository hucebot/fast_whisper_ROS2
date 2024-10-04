# Speech Recognition with Online Whisper

## Testing

### Environment setup

You'll need `sounddevice` and `faster-whisper` which are installable via `pip`, and so could be installed in a simple venv.
However, you'll also need `cudatoolkit`, `cudnn` and `libcublas`.

If these NVIDIA dependencies are not installed on your computer, the fastest and easiest way to test the script is to use `conda` with the provided environment file: `conda env create -f environment.yml`

:warning: this section is a little old, and the script now also use `rclpy` which is not in the environment.
We should split the whisper_online_mic.py into a base script, a script to directly test from the microphone, and a script with the ROS node.

### Script

```
usage: whisper_online_mic.py [-h]
                             [--model {tiny.en,tiny,base.en,base,small.en,small,medium.en,medium,large-v1,large-v2,large-v3,large}]
                             [--model_cache_dir MODEL_CACHE_DIR]
                             [--model_dir MODEL_DIR] [--lang LANG]
                             [--buffer_trimming_sec BUFFER_TRIMMING_SEC]
                             [--microphone_blocksize_sec MICROPHONE_BLOCKSIZE_SEC]
                             [--microphone_source MICROPHONE_SOURCE]
                             [-l {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
                             [--ros-topic ROS_TOPIC]
                             [--max-void-count MAX_VOID_COUNT]

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
  --translate
                        Explicity activate the translation to the target language (defined by --lang).
```

You can discover the device name of your microphone using `sounddevice.query_devices()`.

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


