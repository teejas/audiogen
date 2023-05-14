# audiogen
scripts for configuring a VM w GPU attached to use AudioLDM to generate audio given a text or audio prompt

## gcloud commands

some gcloud commands are required to install the nvidia drivers, ssh to the instance, etc

- `gcloud compute ssh $instance_name --zone $zone --project $project_id --ssh-flag="-L 3000:localhost:3000"` to port-forward, remove `--ssh-flag` to just ssh
  - similarly `gcloud compute scp...` to scp to gcloud instance
- install nvidia drivers: https://cloud.google.com/compute/docs/gpus/install-drivers-gpu
  -  `curl https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install_gpu_driver.py --output install_gpu_driver.py`
  -  `python3 install_gpu_driver.py`

## audioLDM ckpt files
```
ckpt_urls = {
  "audioldm-s-full": "https://zenodo.org/record/7600541/files/audioldm-s-full",
  "audioldm-l-full": "https://zenodo.org/record/7698295/files/audioldm-full-l.ckpt",
  "audioldm-s-full-v2": "https://zenodo.org/record/7698295/files/audioldm-full-s-v2.ckpt",
  "audioldm-m-text-ft": "https://zenodo.org/record/7813012/files/audioldm-m-text-ft.ckpt",
  "audioldm-s-text-ft": "https://zenodo.org/record/7813012/files/audioldm-s-text-ft.ckpt",
  "audioldm-m-full": "https://zenodo.org/record/7813012/files/audioldm-m-full.ckpt"
}
```
