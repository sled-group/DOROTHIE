#!/usr/bin/python3

# ==============================================================================
# -- COPYRIGHT -----------------------------------------------------------------
# ==============================================================================

# Copyright (c) 2014-2017, Anthony Zhang <azhang9@gmail.com>
# All rights reserved.
#
# Copyright (c) 2022 SLED Lab, EECS, University of Michigan.
# authors: Martin Ziqiao Ma (marstin@umich.edu),
#          Ben VanDerPloeg (bensvdp@umich.edu),
#          Owen Yidong Huang (owenhji@umich.edu),
#          Elina Eui-In Kim (euiink@umich.edu),
#
# For a copy, see <https://github.com/Uberi/speech_recognition/blob/master/LICENSE.txt>.


# ==============================================================================
# -- IMPORTS -------------------------------------------------------------------
# ==============================================================================

import socket
import sys
import speech_recognition as sr
import pygame
from gtts import gTTS
import threading
from queue import Queue
import time
import math
import collections
import audioop

SAMPLE_RATE = 48000
CHANNELS = 1
CHUNK = 1024


class CustomRecognizer(sr.Recognizer):
    def __init__(self):
        super(CustomRecognizer, self).__init__()

    def listen_till_signal(self, source, q, timeout=None):
        """
        Records a single phrase from ``source`` (an ``AudioSource`` instance) into an ``AudioData`` instance,
        which it returns.

        This is done by waiting until the audio has an energy above ``recognizer_instance.energy_threshold``
        (the user has started speaking), and then recording until it encounters ``recognizer_instance.pause_threshold``
        seconds of non-speaking or there is no more audio input. The ending silence is not included.
        The ``timeout`` parameter is the maximum number of seconds that this will wait for a phrase to start
        before giving up and throwing an ``speech_recognition.WaitTimeoutError`` exception. If ``timeout`` is ``None``,
        there will be no wait timeout.

        The ``phrase_time_limit`` parameter is the maximum number of seconds that this will allow a phrase to continue
        before stopping and returning the part of the phrase processed before the time limit was reached.
        The resulting audio will be the phrase cut off at the time limit. If ``phrase_timeout`` is ``None``,
        there will be no phrase time limit.

        The ``snowboy_configuration`` parameter allows integration with `Snowboy <https://snowboy.kitt.ai/>`__,
        an offline, high-accuracy, power-efficient hotword recognition engine. When used, this function will pause
        until Snowboy detects a hotword, after which it will unpause. This parameter should either be ``None`` to
        turn off Snowboy support, or a tuple of the form ``(SNOWBOY_LOCATION, LIST_OF_HOT_WORD_FILES)``,
        where ``SNOWBOY_LOCATION`` is the path to the Snowboy root directory, and ``LIST_OF_HOT_WORD_FILES`` is
        a list of paths to Snowboy hotword configuration files (`*.pmdl` or `*.umdl` format).

        This operation will always complete within ``timeout + phrase_timeout`` seconds if both are numbers,
        either by returning the audio data, or by raising a ``speech_recognition.WaitTimeoutError`` exception.
        """

        assert isinstance(source, sr.AudioSource), "Source must be an audio source"
        assert source.stream is not None, "Audio source must be entered before listening, " \
                                          "see documentation for ``AudioSource``; " \
                                          "are you using ``source`` outside of a ``with`` statement?"
        assert self.pause_threshold >= self.non_speaking_duration >= 0

        seconds_per_buffer = float(source.CHUNK) / source.SAMPLE_RATE

        # number of buffers of non-speaking audio during a phrase, before the phrase should be considered complete
        pause_buffer_count = int(math.ceil(self.pause_threshold / seconds_per_buffer))

        # maximum number of buffers of non-speaking audio to retain before and after a phrase
        non_speaking_buffer_count = int(math.ceil(self.non_speaking_duration / seconds_per_buffer))

        # read audio input for phrases until there is a phrase that is long enough
        elapsed_time = 0  # number of seconds of audio read
        buffer = b""  # an empty buffer means that the stream has ended and there is no data left to read
        frames = collections.deque()

        # store audio input until the phrase starts
        while True:

            # handle waiting too long for phrase by raising an exception
            elapsed_time += seconds_per_buffer
            if timeout and elapsed_time > timeout:
                raise sr.WaitTimeoutError("listening timed out while waiting for phrase to start")

            buffer = source.stream.read(source.CHUNK)
            if len(buffer) == 0: break  # reached end of the stream
            frames.append(buffer)
            if len(frames) > non_speaking_buffer_count:  # ensure we only keep the needed amount of non-speaking buffers
                frames.popleft()

            # detect whether speaking has started on audio input
            energy = audioop.rms(buffer, source.SAMPLE_WIDTH)  # energy of the audio signal
            if energy > self.energy_threshold: break

            # dynamically adjust the energy threshold using asymmetric weighted average
            if self.dynamic_energy_threshold:

                # account for different chunk sizes and rates
                damping = self.dynamic_energy_adjustment_damping ** seconds_per_buffer
                target_energy = energy * self.dynamic_energy_ratio
                self.energy_threshold = self.energy_threshold * damping + target_energy * (1 - damping)

        # read audio input until the phrase ends
        pause_count, phrase_count = 0, 0
        phrase_start_time = elapsed_time
        while True:
            # handle phrase being too long by cutting off the audio
            elapsed_time += seconds_per_buffer

            buffer = source.stream.read(source.CHUNK)
            if len(buffer) == 0: break  # reached end of the stream
            frames.append(buffer)
            phrase_count += 1

            # check if speaking has stopped for longer than the pause threshold on the audio input
            energy = audioop.rms(buffer, source.SAMPLE_WIDTH)  # unit energy of the audio signal within the buffer
            if energy > self.energy_threshold:
                pause_count = 0
            else:
                pause_count += 1
            if pause_count > pause_buffer_count:  # end of the phrase
                pass #break
            if not q.empty() and q.get() == 'stop':
                break

        # obtain frame data
        # remove extra non-speaking frames at the end
        for i in range(pause_count - non_speaking_buffer_count):
            frames.pop()
        frame_data = b"".join(frames)

        return sr.AudioData(frame_data, source.SAMPLE_RATE, source.SAMPLE_WIDTH)


def run(cmd_queue, inbox, outbox, session_id):
    pygame.mixer.init()

    def recognizeLoop(outbox, cmd_queue):
        r = CustomRecognizer()

        print('Adjusting for ambient noise...')
        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source)

        while 1:
            try:
                while cmd_queue.get() != 'start': time.sleep(0.1)

                print('Listening...')

                with sr.Microphone() as source:
                    audio = r.listen_till_signal(source, cmd_queue)

                res = r.recognize_google(audio)
                outbox.put(res)
                print('You: %s' % res)

                with open('dorothylogs/log_%d/dorothy_audio/utterance_%d.flac' % (session_id, time.time()), 'wb') as audiofile:
                    audiofile.write(audio.get_flac_data())
            except Exception as e:
                print('Exception in recognize loop: ', str(e))

    recthread = threading.Thread(target=recognizeLoop, args=(outbox, cmd_queue), daemon=True)
    recthread.start()

    while 1:
        textIn = inbox.get()
        print('Them: %s' % textIn)
        tts = gTTS(textIn, lang='en')
        tts.save('temp.mp3')

        pygame.mixer.music.load('temp.mp3')
        pygame.mixer.music.play()


if __name__ == '__main__':
    q = Queue()
    r = threading.Thread(target=run, args=(q,), daemon=True)
    r.start()

    time.sleep(2)
    while 1:
        input('Press enter to begin speaking')
        q.put('start')
        input('Press enter to end speaking')
        q.put('stop')
