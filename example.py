'''
Example Snippets
'''
import utils.utils as utils

## audio path to load
audio_path = "input/Achievements_of_the_Democratic_Party_(Homer_S._Cummings).ogg"


## load full audio in memory (not recommended)
audio_reader = utils.AudioReader(audio_path=audio_path, sr=16000, dtype="float32")
data, sr = audio_reader.read()
# do whatever with data
print(data)


## load audio data as streaming
stream = utils.AudioStreaming(audio_path=audio_path, 
                            blocksize=16000*2, 
                            overlap=0, 
                            padding=0, 
                            sr=16000, 
                            dtype="float32")
for block in stream.generator():
    # process here
    print(len(block))

## microphone streaming 
with utils.MicrophoneStreaming() as stream:
    for block in stream.generator():
        # process data here
        print(len(block))


## saving recording to audio file
filename = "hello.wav" 
with utils.MicrophoneStreaming() as stream:
    stream.record_to_file(filename, duration=10)

# ~~~~~~ Async examples ~~~~~~~~

## saving recording to audio file
filename = "hello.wav" 
async def save_audio(filename, duration):
    async with utils.MicrophoneStreaming() as stream:
        await stream.record_to_file(filename, duration=duration)
print("start recording")
asyncio.run(save_audio(filename=filename, duration=10))


## async microphone streaming
async def capture():
    async with utils.MicrophoneStreaming() as stream:
        async for block in stream.generator():
            # process data here
            print(len(block))
try:
    asyncio.run(capture())
except KeyboardInterrupt:
    print("Exited")
