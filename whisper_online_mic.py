#!/usr/bin/env python3
import time
import queue
import threading

from faster_whisper import WhisperModel
import sounddevice as sd
import numpy as np

import rclpy
from rclpy.logging import LoggingSeverity
from std_msgs.msg import String
from rclpy.node import Node

ROS_LOG_LEVELS = {
    "DEBUG": LoggingSeverity.DEBUG,
    "INFO": LoggingSeverity.INFO,
    "WARN": LoggingSeverity.WARN,
    "ERROR": LoggingSeverity.ERROR,
    "FATAL": LoggingSeverity.FATAL,
}

MICROPHONE_SR = 48000
TARGET_SR = 16000
NODE_NAME = "whisper_asr"


class FasterWhisperASR:
    """Uses faster-whisper library as the backend. Works much faster, appx 4-times (in offline mode). For GPU, it requires installation with a specific CUDNN version."""

    sep = ""

    def __init__(
        self,
        logger,
        lan="auto",
        translate=False,
        modelsize=None,
        cache_dir=None,
        model_dir=None,
    ):
        self.transcribe_kargs = {
            "vad_filter": True,
            "vad_parameters": dict(min_silence_duration_ms=500),
        }
        if translate:
            self.transcribe_kargs["task"] = "translate"
        if lan == "auto":
            self.original_language = None
        else:
            self.original_language = lan
        self.logger = logger
        self.model = self.__load_model(modelsize, cache_dir, model_dir)

    def __load_model(self, modelsize=None, cache_dir=None, model_dir=None):
        #        logging.getLogger("faster_whisper").setLevel(logger.level)
        if model_dir is not None:
            self.logger.debug(
                f"Loading whisper model from model_dir {model_dir}. modelsize and cache_dir parameters are not used."
            )
            model_size_or_path = model_dir
        elif modelsize is not None:
            model_size_or_path = modelsize
        else:
            raise ValueError("modelsize or model_dir parameter must be set")

        # this worked fast and reliably on NVIDIA L40
        model = WhisperModel(
            model_size_or_path,
            device="cuda",
            compute_type="float16",
            download_root=cache_dir,
        )

        # or run on GPU with INT8
        # tested: the transcripts were different, probably worse than with FP16, and it was slightly (appx 20%) slower
        # model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")

        # or run on CPU with INT8
        # tested: works, but slow, appx 10-times than cuda FP16
        #        model = WhisperModel(modelsize, device="cpu", compute_type="int8") #, download_root="faster-disk-cache-dir/")
        return model

    def transcribe(self, audio, init_prompt=""):
        # tested: beam_size=5 is faster and better than 1 (on one 200 second document from En ESIC, min chunk 0.01)
        segments, info = self.model.transcribe(
            audio,
            language=self.original_language,
            initial_prompt=init_prompt,
            beam_size=5,
            word_timestamps=True,
            condition_on_previous_text=True,
            **self.transcribe_kargs,
        )
        self.logger.debug(f"[transcribe info] {info}")

        return list(segments)

    def ts_words(self, segments):
        o = []
        for segment in segments:
            for word in segment.words:
                # not stripping the spaces -- should not be merged with them!
                w = word.word
                t = (word.start, word.end, w)
                o.append(t)
        return o

    def segments_end_ts(self, res):
        return [s.end for s in res]


class HypothesisBuffer:
    def __init__(self, logger):
        self.commited_in_buffer = []
        self.buffer = []
        self.new = []
        self.logger = logger
        self.last_commited_time = 0
        self.last_commited_word = None

    def insert(self, new, offset):
        # compare self.commited_in_buffer and new. It inserts only the words in new that extend the commited_in_buffer, it means they are roughly behind last_commited_time and new in content
        # the new tail is added to self.new

        new = [(a + offset, b + offset, t) for a, b, t in new]
        self.new = [(a, b, t)
                    for a, b, t in new if a > self.last_commited_time - 0.1]

        if len(self.new) >= 1:
            a, b, t = self.new[0]
            if abs(a - self.last_commited_time) < 1:
                if self.commited_in_buffer:
                    # it's going to search for 1, 2, ..., 5 consecutive words (n-grams) that are identical in commited and new. If they are, they're dropped.
                    cn = len(self.commited_in_buffer)
                    nn = len(self.new)
                    for i in range(1, min(min(cn, nn), 5) + 1):  # 5 is the maximum
                        c = " ".join(
                            [self.commited_in_buffer[-j][2] for j in range(1, i + 1)][
                                ::-1
                            ]
                        )
                        tail = " ".join(self.new[j - 1][2]
                                        for j in range(1, i + 1))
                        if c == tail:
                            words = []
                            for j in range(i):
                                words.append(repr(self.new.pop(0)))
                            words_msg = " ".join(words)
                            self.logger.debug(
                                f"removing last {i} words: {words_msg}")
                            break

    def flush(self):
        # returns commited chunk = the longest common prefix of 2 last inserts.

        commit = []
        while self.new:
            na, nb, nt = self.new[0]

            if len(self.buffer) == 0:
                break

            if nt == self.buffer[0][2]:
                commit.append((na, nb, nt))
                self.last_commited_word = nt
                self.last_commited_time = nb
                self.buffer.pop(0)
                self.new.pop(0)
            else:
                break
        self.buffer = self.new
        self.new = []
        self.commited_in_buffer.extend(commit)
        return commit

    def pop_commited(self, time):
        while self.commited_in_buffer and self.commited_in_buffer[0][1] <= time:
            self.commited_in_buffer.pop(0)

    def complete(self):
        return self.buffer


class OnlineASRProcessor:
    SAMPLING_RATE = 16000

    def __init__(self, asr, logger, buffer_trimming_sec=15):
        """asr: WhisperASR object
        tokenizer: sentence tokenizer object for the target language. Must have a method *split* that behaves like the one of MosesTokenizer. It can be None, if "segment" buffer trimming option is used, then tokenizer is not used at all.
        ("segment", 15)
        buffer_trimming: a pair of (option, seconds), where option is either "sentence" or "segment", and seconds is a number. Buffer is trimmed if it is longer than "seconds" threshold. Default is the most recommended option.
        """
        self.asr = asr
        self.buffer_trimming_sec = buffer_trimming_sec
        self.logger = logger
        self.init()

    def init(self):
        """run this when starting or restarting processing"""
        self.audio_buffer = np.array([], dtype=np.float32)
        self.buffer_time_offset = 0

        self.transcript_buffer = HypothesisBuffer(self.logger)
        self.commited = []

    def insert_audio_chunk(self, audio):
        self.audio_buffer = np.append(self.audio_buffer, audio)

    def prompt(self):
        """Returns a tuple: (prompt, context), where "prompt" is a 200-character suffix of commited text that is inside of the scrolled away part of audio buffer.
        "context" is the commited text that is inside the audio buffer. It is transcribed again and skipped. It is returned only for debugging and logging reasons.
        """
        k = max(0, len(self.commited) - 1)
        while k > 0 and self.commited[k - 1][1] > self.buffer_time_offset:
            k -= 1

        p = self.commited[:k]
        p = [t for _, _, t in p]
        prompt = []
        length = 0
        while p and length < 200:  # 200 characters prompt size
            x = p.pop(-1)
            length += len(x) + 1
            prompt.append(x)
        non_prompt = self.commited[k:]
        return self.asr.sep.join(prompt[::-1]), self.asr.sep.join(
            t for _, _, t in non_prompt
        )

    def process_iter(self):
        """Runs on the current audio buffer.
        Returns: a tuple (beg_timestamp, end_timestamp, "text"), or (None, None, "").
        The non-emty text is confirmed (committed) partial transcript.
        """

        prompt, non_prompt = self.prompt()
        self.logger.debug(f"PROMPT: {prompt}")
        self.logger.debug(f"CONTEXT: {non_prompt}")
        self.logger.debug(
            f"transcribing {len(self.audio_buffer)/self.SAMPLING_RATE:2.2f} seconds from {self.buffer_time_offset:2.2f}"
        )
        res = self.asr.transcribe(self.audio_buffer, init_prompt=prompt)

        # transform to [(beg,end,"word1"), ...]
        tsw = self.asr.ts_words(res)

        self.transcript_buffer.insert(tsw, self.buffer_time_offset)
        o = self.transcript_buffer.flush()
        self.commited.extend(o)
        completed = self.to_flush(o)
        self.logger.debug(f">>>>COMPLETE NOW: {completed}")
        the_rest = self.to_flush(self.transcript_buffer.complete())
        self.logger.debug(f"INCOMPLETE: {the_rest}")

        if len(self.audio_buffer) / self.SAMPLING_RATE > self.buffer_trimming_sec:
            self.chunk_completed_segment(res)
            self.logger.debug("chunking segment")

        self.logger.debug(
            f"len of buffer now: {len(self.audio_buffer)/self.SAMPLING_RATE:2.2f}"
        )
        return self.to_flush(o)

    def chunk_completed_segment(self, res):
        if self.commited == []:
            return

        ends = self.asr.segments_end_ts(res)

        t = self.commited[-1][1]

        if len(ends) > 1:
            e = ends[-2] + self.buffer_time_offset
            while len(ends) > 2 and e > t:
                ends.pop(-1)
                e = ends[-2] + self.buffer_time_offset
            if e <= t:
                self.logger.debug(f"--- segment chunked at {e:2.2f}")
                self.chunk_at(e)
            else:
                self.logger.debug("--- last segment not within commited area")
        else:
            self.logger.debug("--- not enough segments to chunk")

    def chunk_at(self, time):
        """trims the hypothesis and audio buffer at "time" """
        self.transcript_buffer.pop_commited(time)
        cut_seconds = time - self.buffer_time_offset
        self.audio_buffer = self.audio_buffer[int(
            cut_seconds * self.SAMPLING_RATE):]
        self.buffer_time_offset = time

    def finish(self):
        """Flush the incomplete text when the whole processing ends.
        Returns: the same format as self.process_iter()
        """
        o = self.transcript_buffer.complete()
        f = self.to_flush(o)
        self.logger.debug(f"last, noncommited: {f}")
        return f

    def to_flush(
        self,
        sents,
        sep=None,
        offset=0,
    ):
        # concatenates the timestamped words or sentences into one sequence that is flushed in one line
        # sents: [(beg1, end1, "sentence1"), ...] or [] if empty
        # return: (beg1,end-of-last-sentence,"concatenation of sentences") or (None, None, "") if empty
        if sep is None:
            sep = self.asr.sep
        t = sep.join(s[2] for s in sents)
        if len(sents) == 0:
            b = None
            e = None
        else:
            b = offset + sents[0][0]
            e = offset + sents[-1][1]
        return (b, e, t)


def init_online_whisper(args, logger):
    """
    Creates and configures an ASR and ASR Online instance based on the specified backend and arguments.
    """
    # Init FasterWhisperASR
    t = time.time()
    if args.model_dir is not None:
        logger.info(
            f"Loading Whisper from {args.model_dir} for {args.lang}...")
    else:
        logger.info(f"Loading Whisper {args.model} model for {args.lang}...")
    asr = FasterWhisperASR(
        logger,
        translate=args.translate,
        lan=args.lang,
        modelsize=args.model,
        cache_dir=args.model_cache_dir,
        model_dir=args.model_dir,
    )
    e = time.time()
    logger.info(f"done. It took {round(e-t,2)} seconds.")

    # Warm up Whisper
    t = time.time()
    logger.info("Warming up Whisper with random data")
    asr.transcribe(np.random.rand(TARGET_SR * 30).astype(np.float32))
    e = time.time()
    logger.info(f"done. It took {round(e-t,2)} seconds.")

    # Return the online wrapper
    return OnlineASRProcessor(asr, logger, buffer_trimming_sec=args.buffer_trimming_sec)


class WhisperPublisher(Node):
    def __init__(self, args):
        super().__init__(NODE_NAME)
        self.get_logger().info("Whisper publisher is starting !")

        self.args = args

        self.publisher = self.create_publisher(String, args.ros_topic, 10)

        self.thread = threading.Thread(target=self.pub)
        self.thread.start()

    def pub(self):
        comp_q = queue.Queue()
        online = init_online_whisper(self.args, self.get_logger())
        sentence_buffer = ""
        void_count = 0

        def asr_callback(indata, frames, time, status):
            online.insert_audio_chunk(indata[::3].squeeze())
            comp_q.put(True)

        with sd.InputStream(
            device=self.args.microphone_source,
            dtype=np.float32,
            channels=1,
            samplerate=MICROPHONE_SR,  # I have issue using 16kHz with the Wireless Go II microphone
            blocksize=int(self.args.microphone_blocksize_sec * MICROPHONE_SR),
            callback=asr_callback,
        ):
            while comp_q.get():
                try:
                    o = online.process_iter()
                except AssertionError as e:
                    self.get_logger().error(f"assertion error: {e}")
                else:
                    sentence_buffer += o[2]
                    self.get_logger().debug(
                        f"ASR sentence buffer >> {sentence_buffer}")

                    if o[2] == "":
                        void_count += 1
                    else:
                        void_count = 0

                    if (
                        sentence_buffer.endswith((".", "!", "?"))
                        or void_count > self.args.max_void_count
                    ) and sentence_buffer != "":
                        msg = String()
                        msg.data = sentence_buffer
                        self.publisher.publish(msg)
                        self.get_logger().info(f"ASR msg >> {sentence_buffer}")
                        sentence_buffer = ""


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="large-v2",
        choices="tiny.en,tiny,base.en,base,small.en,small,medium.en,medium,large-v1,large-v2,large-v3,large".split(
            ","
        ),
        help="Name size of the Whisper model to use (default: large-v2). The model is automatically downloaded from the model hub if not present in model cache dir.",
    )
    parser.add_argument(
        "--model_cache_dir",
        type=str,
        default=None,
        help="Overriding the default model cache dir where models downloaded from the hub are saved",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="Dir where Whisper model.bin and other files are saved. This option overrides --model and --model_cache_dir parameter.",
    )
    parser.add_argument(
        "--lang",
        "--language",
        type=str,
        default="auto",
        help="Source language code, e.g. en,de,cs, or 'auto' for language detection.",
    )
    parser.add_argument(
        "--buffer_trimming_sec",
        type=float,
        default=15,
        help="Buffer trimming length threshold in seconds. If buffer length is longer, trimming sentence/segment is triggered.",
    )
    parser.add_argument(
        "--microphone_blocksize_sec",
        type=float,
        default=0.01,
        help="Size of the audio block, default to 10ms.",
    )
    parser.add_argument(
        "--microphone_source",
        type=str,
        default="Wireless GO II RX",
        help="Device name to use.",
    )
    parser.add_argument(
        "-l",
        "--log-level",
        dest="log_level",
        choices=["DEBUG", "INFO", "WARN", "ERROR", "FATAL"],
        help="Set the log level",
        default="INFO",
    )
    parser.add_argument(
        "--ros-topic",
        dest="ros_topic",
        type=str,
        default="eurobin/asr",
        help="ROS topic on with publish recognized sentences.",
    )
    parser.add_argument(
        "--max-void-count",
        type=int,
        default=5,
        help="Maximum number of consecutive void recognition before sending the sentence buffer.",
    )
    parser.add_argument(
        "--translate",
        action="store_true",
        default=False,
        help="Explicity activate the translation to the target language (defined by --lang).",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    rclpy.logging.set_logger_level(NODE_NAME, ROS_LOG_LEVELS[args.log_level])

    rclpy.init()
    rclpy.spin(WhisperPublisher(args))
    rclpy.shutdown()
