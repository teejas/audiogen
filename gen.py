import os
import scipy.io.wavfile as wavf
import torch
import soundfile as sf
import argparse
from audioldm import text_to_audio, style_transfer, super_resolution_and_inpainting, build_model, latent_diffusion

ckpt_urls = {
  "audioldm-s-full": "https://zenodo.org/record/7600541/files/audioldm-s-full.ckpt",
  "audioldm-full-l": "https://zenodo.org/record/7698295/files/audioldm-full-l.ckpt",
  "audioldm-full-s-v2": "https://zenodo.org/record/7698295/files/audioldm-full-s-v2.ckpt",
  "audioldm-m-text-ft": "https://zenodo.org/record/7813012/files/audioldm-m-text-ft.ckpt",
  "audioldm-s-text-ft": "https://zenodo.org/record/7813012/files/audioldm-s-text-ft.ckpt",
  "audioldm-m-full": "https://zenodo.org/record/7813012/files/audioldm-m-full.ckpt"
}

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

def setup_args() -> (str, str, str):
  parser = argparse.ArgumentParser(
                    prog='audiogen',
                    description='generates audio using AudioLDM and writes it to out.wav',
                    epilog='~tejas')
  parser.add_argument('-c', '--checkpoint', default="audioldm-m-full", 
                      choices=["audioldm-s-full", "audioldm-full-l", "audioldm-full-s-v2", "audioldm-m-text-ft", "audioldm-s-text-ft", "audioldm-m-full"])
  parser.add_argument('-i', '--input_file', required=True)
  parser.add_argument('-o', '--output_file', default="out.wav")

  args = parser.parse_args()
  return (args.checkpoint, args.input_file, args.output_file)


if __name__ == '__main__':
  (use_checkpoint, input_file, output_file) = setup_args()
  ckpt_path = "./ckpt/" + use_checkpoint + ".ckpt"
  if not os.path.exists(ckpt_path):
    r = requests.get(ckpt_urls[use_checkpoint], allow_redirects=True)
    open(ckpt_path, 'wb').write(r.content)
  audioldm = build_model(ckpt_path=ckpt_path, model_name=use_checkpoint)
  sr = 16000 # sample rate

  generated_audio = text2audio('placeholder', duration=20, audio_path=input_file, guidance_scale=7, random_seed=0, n_candidates=3, steps=200)
  sf.write(output_file, generated_audio.T, sr, subtype='PCM_24')
