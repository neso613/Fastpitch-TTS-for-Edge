
from nemo.collections.tts.models import FastPitchModel
from nemo.collections.tts.models.base import Vocoder
import soundfile as sf

# Load FastPitchModel
spec_generator = FastPitchModel.from_pretrained("tts_en_fastpitch")
#spec_generator = FastPitchModel.restore_from("tts_en_fastpitch_align.nemo")

# Load vocoder
model = Vocoder.from_pretrained(model_name="tts_hifigan")
#model = Vocoder.restore_from("tts_hifigan.nemo")


parsed = spec_generator.parse("You can type your sentence here to get nemo to produce speech.")
spectrogram = spec_generator.generate_spectrogram(tokens=parsed)
audio = model.convert_spectrogram_to_audio(spec=spectrogram)[0]

# Save the audio to disk in a file called speech.wav
sf.write("speech.wav", audio.detach().numpy(), 22050)
