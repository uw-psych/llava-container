# llava-container

This container provides a convenient way to run [LLaVA](https://github.com/haotian-liu/LLaVA) on Hyak.

## Running LLaVA on Hyak 🍇

First, you'll need to log in to Hyak. If you've never set this up, go [here](https://uw-psych.github.io/compute_docs).

```bash
ssh your-uw-netid@klone.hyak.uw.edu
```

Then, you'll need to request a compute node. You can do this with the `salloc` command:

```bash
# Request a GPU node with 8 CPUs, 2 GPUs, 64GB of RAM, and 1 hour of runtime:
# (Note: you may need to change the account and partition)
salloc --account escience --partition gpu-a40 --mem 64G -c 8 --time 1:00:00 --gpus 2
```

One you're logged in to the compute node, you should set up your cache directories and Apptainer settings.

👉 *If you're following this tutorial, **you should do this every time you're running LLaVA on Hyak!** This is because the default settings for Apptainer will use your home directory for caching, which will quickly fill up your home directory and cause your jobs to fail.*

```bash
# Do this in every session where you're running LLaVA on Hyak!

# Set up cache directories:
export APPTAINER_CACHEDIR="/gscratch/scrubbed/${USER}/.cache/apptainer"
export HUGGINGFACE_HUB_CACHE="/gscratch/scrubbed/${USER}/.cache/huggingface"
mkdir -p "${APPTAINER_CACHEDIR}" "${HUGGINGFACE_HUB_CACHE}"

# Set up Apptainer:
export APPTAINER_BIND=/gscratch APPTAINER_WRITABLE_TMPFS=1 APPTAINER_NV=1
```

Then, you can run LLaVA. Let's try with the sample image on LLaVA's repository:

![Sample image](https://llava-vl.github.io/static/images/view.jpg)


```bash
# Run LLaVA:
apptainer run \
    oras://ghcr.io/maouw/llava-container/llava-container:latest \
    llava-run \
    --model-path liuhaotian/llava-v1.5-7b \
    --image-file "https://llava-vl.github.io/static/images/view.jpg" \
    --query "What's going on here?"

# Description of the arguments:
# llava-run: the command to run in the container
# --model-path: the name of the model to use
# --image-file: the URL of the image to use
# --query: what to ask the model
```

If it's working, you should see output that looks something like this:

> The image features a pier extending out into a large body of water, possibly a lake or a river. The pier is made of wood and has a few benches placed on it, providing a place for people to sit and enjoy the view. The water appears calm and serene, making it an ideal spot for relaxation and contemplation.
>
> In the background, there are mountains visible, adding to the picturesque scenery. The pier is situated in front of a forest, creating a peaceful and natural atmosphere.

When you're done, you can exit the compute node with the command `exit` or `Ctrl-D`.

### Chat mode 🗣️

For chat, just pass `--chat` instead of `--query`:

```bash
apptainer run \
    oras://ghcr.io/maouw/llava-container/llava-container:latest \
    llava-run \
    --model-path liuhaotian/llava-v1.5-7b \
    --image-file "https://llava-vl.github.io/static/images/view.jpg" \
    --chat
```

### Running other commands 🏃

If you want to a different command, such as one of the commands that comes with LLaVA, you can pass it after the image name:

```bash
apptainer run \
    oras://ghcr.io/maouw/llava-container/llava-container:latest \
    python -m llava.serve.cli
```

### Improving startup time 🚀

If you notice slowness when launching the container, you can try extracting the container image to a sandbox directory:

```bash
# Set up a sandbox directory:
SANDBOX="/tmp/${USER}/sandbox/llava" && mkdir -p "$(dirname "${SANDBOX}")"

# Extract the container image to the sandbox:
apptainer build --sandbox "${SANDBOX}"  oras://ghcr.io/maouw/llava-container/llava-container:latest

# Run LLaVA by passing the sandbox directory instead of the image URL:
apptainer run \
    "${SANDBOX}" \
    llava-run \
    --model-path liuhaotian/llava-v1.5-7b \
    --image-file "https://llava-vl.github.io/static/images/view.jpg" \
    --query "What's going on here?"
```

### Running the web interface 🕸️

Included in the container is a wrapper script for the LLaVA web interface. To run it, you can use the following command:

```bash
apptainer run \
    oras://ghcr.io/maouw/llava-container/llava-container:latest \
    hyak-llava-web
```

This script will print out a command to set up an SSH tunnel to the web interface. You can then open the web interface by visiting `http://localhost:8000` in your browser. The output should look something like this:

```bash
# To access the gradio web server, run the following command on your local machine:                                   
ssh -o StrictHostKeyChecking=no -N -L 8000:localhost:53641 -J altan@klone.hyak.uw.edu altan@g3021
```

You should be able to copy and paste this command into your terminal to set up the SSH tunnel. Then, you can open `http://localhost:8000` in your browser to access the web interface.

To configure the web interface, you can set the following environment variables:

- `MODEL_PATHS`: a list of model paths, quoted and separated by space (default: "liuhaotian/llava-v1.5-7b")
  - Available models include, but are not limited to:
    - liuhaotian/llava-v1.5-7b
    - liuhaotian/llava-v1.5-13b
    - liuhaotian/llava-v1.5-7b-lora
    - liuhaotian/llava-v1.5-13b-lora

  See https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md for more details.

- `GRADIO_CONTROLLER_PORT`: the port number for the gradio controller (or leave it empty to use a random port)
- `LOCAL_HTTP_PORT`: the port number to print for the local HTTP server SSH tunnel command (default: 8000)

For example:

```bash
export MODEL_PATHS='liuhaotian/llava-v1.5-13b' # Use the 13b model instead of the 7b model
export LOCAL_HTTP_PORT=9000 # Use port 9000 instead of 8000
apptainer run \
    oras://ghcr.io/maouw/llava-container/llava-container:latest \
    hyak-llava-web
```

👉 *You need to select the model from the dropdown to start. If the model doesn't appear in the dropdown, wait a few seconds and refresh the page.*

## `llava-run`

The `llava-run.py` script is a modification of [`LLaVA/lava/eval/run_llava.py`](https://github.com/haotian-liu/LLaVA/blob/main/llava/eval/run_llava.py) that adds support for loading 4- and 8-bit models as found in [`LaVA/llava/serve/cli.py`](https://github.com/haotian-liu/LLaVA/blob/main/llava/serve/cli.py), as well as a chat mode that allows you to have a conversation with the model.

The following describes the usage of `llava-run`:

```plain
This container provides a convenient way to run LLaVA. In addition to the LLaVA
module, it includes the commands:
  - `llava-run`, a command-line wrapper for LLaVA inference
  - `hyak-llava-web`, a wrapper to launch the gradio web interface and issue an
    SSH connection string you can copy to open a tunnel to your own computer.
 
To run LLaVA with the `llava-run` script, use the following command:
  apptainer run --nv --writable-tmpfs \
    oras://ghcr.io/maouw/llava-container/llava-container:latest \
    llava-run [llava-run arguments]

You must pass the "--nv" flag to enable GPU support.

Depending on your intended use, you may also want to pass the "--bind" flag
to mount a directory from the host system into the container.

To specify a directory to use for the HuggingFace model cache and enable access
to /gscratch, use the following command:
  apptainer run --nv --writable-tmpfs \
    --env HUGGINGFACE_HUB_CACHE=/path/to/cache \
    --bind /gscratch \
    oras://ghcr.io/maouw/llava-container/llava-container:latest \
    llava-run [llava-run arguments]


The following describes the usage of this script:

  llava-run [-h] [--model-path MODEL_PATH] [--model-base MODEL_BASE]
    --image-file IMAGE (--query QUERY | --chat) [--conv-mode CONV_MODE]
    [--sep SEP] [--temperature TEMPERATURE] [--top_p N]
    [--num_beams NUM_BEAMS] [--max_new_tokens N]
    [--load-8bit] [--load-4bit] [--device DEVICE]
    [--hf-cache-dir HF_CACHE_DIR]

  options:
    -h, --help            show this help message and exit
    --model-path MODEL_PATH     Model path
    --model-base MODEL_BASE     Model base
    --image-file IMAGE_FILE     Image file or URL
    --query QUERY               Query
	--chat						Chat mode 
    --conv-mode CONV_MODE       Conversation mode
    --sep SEP                   Separator for image files
    --temperature TEMPERATURE   Temperature
    --top_p TOP_P               Top p
    --num_beams NUM_BEAMS       Number of beams
    --max_new_tokens MAX_NEW_TOKENS Max new tokens
    --load-8bit                 Load 8bit model
    --load-4bit                 Load 4bit model
    --device DEVICE             cuda or cpu
    --hf-cache-dir HF_CACHE_DIR HuggingFace cache directory
  
  For details on the arguments, see the LLaVA documentation and the usage infor-
  mation for llava.eval.run_llava and llava.serve.cli.
```

See the [documentation](https://github.com/haotian-liu/LLaVA/blob/main/README.md) for LLaVA or the source code for [`llava/eval/run_llava.py`](https://github.com/haotian-liu/LLaVA/blob/main/llava/eval/run_llava.py) and [`llava/serve/cli.py`](https://github.com/haotian-liu/LLaVA/blob/main/llava/serve/cli.py) for more information on the arguments.
