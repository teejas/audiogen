import os
import scipy.io.wavfile as wavf
import torch
import soundfile as sf
from audioldm import text_to_audio, style_transfer, super_resolution_and_inpainting, build_model, latent_diffusion

def text2audio(text, duration, audio_path, guidance_scale, random_seed, n_candidates, steps):
  waveform = text_to_audio(
    audioldm,
    text,
    audio_path,
    random_seed,
    duration=duration,
    guidance_scale=guidance_scale,
    ddim_steps=steps,
    n_candidate_gen_per_text=int(n_candidates)
  )
  if(len(waveform) == 1):
    waveform = waveform[0]
  return waveform


if __name__ == '__main__':
  use_checkpoint = "audioldm-m-full"
  ckpt_path = "./ckpt/audioldm-m-full.ckpt"

  audioldm = build_model(ckpt_path=ckpt_path, model_name=use_checkpoint)
  sr = 16000 # sample rate

  sample_dir = "samples/"
  song = "iz_over_the_rainbow_shortened.mp3"
  interview = "interview.mp3"
  piano = "short_melody.mp3"
  file_out = "out.wav"
  generated_audio = text2audio('placeholder', duration=20, audio_path=sample_dir+piano, guidance_scale=7, random_seed=0, n_candidates=3, steps=200)
  sf.write(file_out, generated_audio.T, sr, subtype='PCM_24')
