# llava-container

This container provides a convenient way to run [LLaVA](https://github.com/haotian-liu/LLaVA).

To run LLaVA, use the following command:

```bash
apptainer run --nv llava-container.sif
```

You must pass the `--nv` flag to enable GPU support.

Depending on your intended use, you may also want to pass the `--bind` flag to mount a directory from the host system into the container, or add flags like `--contain` or `--cleanenv` to change the container runtime behavior


To specify a directory to use for the HuggingFace model cache, you can pass the `--env` flag to set the `HUGGINGFACE_HUB_CACHE` environment variable. For example:

```bash
apptainer run --nv --env HUGGINGFACE_HUB_CACHE=/path/to/cache llava-container.sif [..args..]
```

 
By default, the container will run the script `llava-run.py` (from this repository) with the arguments provided. 

The `llava-run.py` script is a modification of `LLaVA/lava/eval/run_llava.py` that adds support for loading 4- and 8-bit models as found in `LaVA/llava/serve/cli.py`.

Here is a complete example of launching the container to run LLaVA:

```bash
apptainer run --nv --env HUGGINGFACE_HUB_CACHE=$PWD/hubcache --nv llava-container.sif \
    --model-path liuhaotian/llava-v1.5-7b \
    --image-file "https://llava-vl.github.io/static/images/view.jpg" \
    --load-4bit \
    --query "What's going on here?"
```

The following describes the usage of `llava-run`:

```plain
    llava-run [-h] [--model-path MODEL_PATH] [--model-base MODEL_BASE]
        --image-file IMAGE_FILE --query QUERY [--conv-mode CONV_MODE]
        [--sep SEP] [--temperature TEMPERATURE] [--top_p TOP_P]
        [--num_beams NUM_BEAMS] [--max_new_tokens MAX_NEW_TOKENS]
        [--load-8bit] [--load-4bit] [--device DEVICE]
        [--hf-cache-dir HF_CACHE_DIR]

    options:
        -h, --help                         show this help message and exit
        --model-path MODEL_PATH            Model path
        --model-base MODEL_BASE            Model base
        --image-file IMAGE_FILE            Image file
        --query QUERY                      Query
        --conv-mode CONV_MODE              Conversation mode
        --sep SEP                          Separator for image files
        --temperature TEMPERATURE          Temperature
        --top_p TOP_P                      Top p
        --num_beams NUM_BEAMS              Number of beams
        --max_new_tokens MAX_NEW_TOKENS    Max new tokens
        --load-8bit                        Load 8bit model
        --load-4bit                        Load 4bit model
        --device DEVICE                    cuda or cpu
        --hf-cache-dir HF_CACHE_DIR        HuggingFace cache directory 
```

For details on the arguments, see the LLaVA documentation.
