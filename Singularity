Bootstrap: docker
From: mambaorg/micromamba:{{ MICROMAMBA_TAG }}

%arguments
	MICROMAMBA_TAG=jammy-cuda-12.3.1
	PYTHON_VERSION=3.11
	LLAVA_URL=https://codeload.github.com/haotian-liu/LLaVA/tar.gz/refs/heads/main

%labels
	VERSION 0.0.3

%setup
	[ -n "${APPTAINER_ROOTFS:-}" ] && ./.build-scripts/write-apptainer-labels.sh >"${APPTAINER_ROOTFS}/.build_labels"

%files
	llava-run.py /opt/local/bin/llava-run

%post
	set -ex
	export MAMBA_DOCKERFILE_ACTIVATE=1
	export DEBIAN_FRONTEND=noninteractive
	
	# Set up the micromamba base environment, using requests to download LLaVA:
	micromamba install -y -n base -c conda-forge python={{ PYTHON_VERSION }} pip requests
	micromamba run -n base python -m pip install --upgrade pip
	
	# Download and install LLaVA:
	mkdir -p /opt/setup/llava && cd /opt/setup/llava
	micromamba run -n base python -c \
		'import requests;open("llava.tar.gz", "wb").write(requests.get("{{ LLAVA_URL }}",allow_redirects=True).content)' &&
		tar -xzf llava.tar.gz --strip-components=1
	
	# Update pyproject.toml to include the package data in examples (web server breaks without):
	printf '[tool.setuptools.package-data]\nllava = ["serve/examples/*.jpg"]' >> pyproject.toml

	# Install LLaVA and dependencies:
	micromamba run -n base python -m pip install --no-cache-dir --config-settings="--install-data=$PWD/llava" .
	
	# Clean up:
	micromamba run -n base python -m pip cache purge
	micromamba clean --all --yes
	rm -rf /opt/setup

%environment
	export MAMBA_DOCKERFILE_ACTIVATE=1
	export PATH="/opt/local/bin:${PATH}"

%runscript
	# Run the provided command with the micromamba base environment activated:
	eval "$(micromamba shell hook --shell posix)"
	micromamba activate
	echo "Using HUGGINGFACE_HUB_CACHE=\"${HUGGINGFACE_HUB_CACHE:-}\"" >&2
	exec "$@"

%help
	This container provides a convenient way to run
	[LLaVA](https://github.com/haotian-liu/LLaVA).

	To run LLaVA with the `llava-run` script, use the following command:
		apptainer run --nv --writable-tmpfs llava-container.sif llava-run
	
	You must pass the "--nv" flag to enable GPU support.

	Depending on your intended use, you may also want to pass the "--bind" flag
	to mount a directory from the host system into the container, or add flags
	like "--contain" or "--cleanenv" to change the container runtime behavior.
	
	To specify a directory to use for the HuggingFace model cache, use the
	following command:
		apptainer run -writable-tmpfs \
					   --nv
					   --env HUGGINGFACE_HUB_CACHE=/path/to/cache \
				  llava-container.sif llava-run
	
	This container includes a script called "llava-run" that runs LLaVA with the
	arguments provided. The following describes the usage of this script:

		llava-run [-h] [--model-path MODEL_PATH] [--model-base MODEL_BASE]
			--image-file IMAGE_FILE --query QUERY [--conv-mode CONV_MODE]
			[--sep SEP] [--temperature TEMPERATURE] [--top_p TOP_P]
			[--num_beams NUM_BEAMS] [--max_new_tokens MAX_NEW_TOKENS]
			[--load-8bit] [--load-4bit] [--device DEVICE]
			[--hf-cache-dir HF_CACHE_DIR]

		options:
			-h, --help				show this help message and exit
			--model-path MODEL_PATH Model path
			--model-base MODEL_BASE Model base
			--image-file IMAGE_FILE Image file
			--query QUERY         Query
			--conv-mode CONV_MODE Conversation mode
			--sep SEP             Separator for image files
			--temperature TEMPERATURE Temperature
			--top_p TOP_P         Top p
			--num_beams NUM_BEAMS Number of beams
			--max_new_tokens MAX_NEW_TOKENS
									Max new tokens
			--load-8bit           Load 8bit model
			--load-4bit           Load 4bit model
			--device DEVICE       cuda or cpu
			--hf-cache-dir HF_CACHE_DIR HuggingFace cache directory
		
		For details on the arguments, see the LLaVA documentation.

