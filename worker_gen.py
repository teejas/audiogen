# worker file to generate audio given tasks from redis queue
import redis
import soundfile as sf
from gen import text2audio
from audioldm import text_to_audio, style_transfer, super_resolution_and_inpainting, build_model, latent_diffusion

if __name__ == '__main__':
  r = redis.Redis(host='localhost', port=6379, decode_responses=True)
  while(len(r.keys()) > 0):
    fp = "./samples/" + r.keys()[0]
    cmd = r.get(r.keys()[0])
    print("key: " + fp + ", cmd: " + cmd)
    if cmd == "audiogen":
      audioldm = build_model(ckpt_path="./ckpt/audioldm-m-full.ckpt", model_name="audioldm-m-full")
      sr = 16000 # sample rate

      generated_audio = text2audio(audioldm, 'placeholder', duration=20, audio_path=fp, guidance_scale=7, random_seed=0, n_candidates=3, steps=200)
      sf.write("./samples/generated_audio.wav", generated_audio.T, sr, subtype='PCM_24')
      print("GOING TO GENERATE AUDIO FOR " + fp)
      r.delete(r.keys()[0])