import os
import requests
import scipy.io.wavfile as wavf
import soundfile as sf
import argparse
from tqdm import tqdm
from urllib.request import urlopen
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

def download_ckpt(ckpt_url: str, ckpt_path: str) -> None:
  print("no .ckpt file found, downloading from zenodo...") 
  response = requests.get(ckpt_url, stream=True, allow_redirects=True)
  total_size_in_bytes= int(response.headers.get('content-length', 0))
  block_size = 1024 #1 Kibibyte
  progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
  with open(ckpt_path, 'wb') as f:
    for data in response.iter_content(block_size):
      progress_bar.update(len(data))
      f.write(data)
  progress_bar.close()
  if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
    raise RuntimeError("Failed to download ckpt file.")

def setup_args() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
                    prog='audiogen',
                    description='generates audio using AudioLDM and writes it to out.wav',
                    epilog='~tejas')
  parser.add_argument('-c', '--checkpoint', default="audioldm-m-full", 
                      choices=["audioldm-s-full", "audioldm-full-l", "audioldm-full-s-v2", 
                               "audioldm-m-text-ft", "audioldm-s-text-ft", "audioldm-m-full"])
  parser.add_argument('-i', '--input_file', required=True)
  parser.add_argument('-o', '--output_file', default="out.wav")

  return parser.parse_args()

if __name__ == '__main__':
  args = setup_args()
  ckpt_path = "./ckpt/" + args.checkpoint + ".ckpt"
  if not os.path.exists(ckpt_path):
    download_ckpt(ckpt_urls[args.checkpoint], ckpt_path)

  audioldm = build_model(ckpt_path=ckpt_path, model_name=args.checkpoint)
  sr = 16000 # sample rate

  generated_audio = text2audio('placeholder', duration=20, audio_path=args.input_file, guidance_scale=7, random_seed=0, n_candidates=3, steps=200)
  sf.write(args.output_file, generated_audio.T, sr, subtype='PCM_24')
