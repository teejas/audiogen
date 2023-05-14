# audiogen
scripts for configuring a VM w GPU attached to use AudioLDM to generate audio given a text or audio prompt

## gcloud commands

some gcloud commands are required to install the nvidia drivers, ssh to the instance, etc

- `gcloud compute ssh $instance_name --zone $zone --project $project_id --ssh-flag="-L 3000:localhost:3000"` to port-forward, remove `--ssh-flag` to just ssh
  - similarly `gcloud compute scp...` to scp to gcloud instance
- install nvidia drivers: https://cloud.google.com/compute/docs/gpus/install-drivers-gpu (already in `install.sh`)
  -  `curl https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install_gpu_driver.py --output install_gpu_driver.py`
  -  `python3 install_gpu_driver.py`

