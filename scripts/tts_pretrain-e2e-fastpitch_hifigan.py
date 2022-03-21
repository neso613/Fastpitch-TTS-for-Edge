import soundfile as sf
from nemo.collections.tts.models import FastPitchHifiGanE2EModel


# Load the model from NGC
model = FastPitchHifiGanE2EModel.from_pretrained(model_name="tts_en_e2e_fastpitchhifigan")
#model = FastPitchHifiGanE2EModel.restore_from(restore_path="../path_of_dir/tts_en_e2e_fastpitchhifigan.nemo")
model.eval()

# Run inference
tokens = model.parse("Hey, I can speak!")
audio = model.convert_text_to_waveform(tokens=tokens)[0]

# Save the audio to disk in a file called speech.wav
sf.write("speech.wav", audio.detach().numpy(), 22050)
