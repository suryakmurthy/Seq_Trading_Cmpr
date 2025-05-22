
# ST-CR User Study Website (Dockerized)

This repository contains a Dockerized Flask application used in the user study for the paper:

**Sequential Resource Trading Using Comparison-Based Gradient Estimation**  
Authors: Surya Murthy, Mustafa O. Karabag, and Ufuk Topcu

## Overview

This website implements the trading interface and backend used for the human-agent user studies in the above paper. The backend supports multiple trading algorithms including the proposed ST-CR algorithm and baseline methods. The application is containerized with Docker for easy local deployment and reproducibility.

## Prerequisites

- Docker must be installed and running.
- Clone this repository and navigate into it:

```
git clone <your-repo-url>
cd <your-repo-name>
```

## File Structure

The root directory contains the following files:

- `Dockerfile` – Specifies how the Docker image is built.
- `requirements.txt` – Lists required Python packages (e.g., Flask, OpenAI).
- `website_frontend.py` – Entry point for the Flask server (frontend logic).
- `python_node.py` – Backend handler for request routing and OpenAI integration.
- `algo_STCR.py` – Implements the ST-CR algorithm described in the paper.
- `algo_pure_GPT.py` – Implements a pure GPT-based trading strategy.
- `algo_random.py` – Implements a randomized trading baseline.
- `algo_GCA.py` – Implements the GCA (Greedy Concession Algorithm) baseline.

These algorithm files are imported directly in the main scripts like so:

## Configuration Requirement

To use the OpenAI-based features, **you must provide your own OpenAI API key**.

1. Open `python_node.py`
2. Locate the line (Line 78):

```python
openai.api_key = None  # Insert Your API Key Here
```

3. Replace `None` with your actual API key:

```python
openai.api_key = "sk-..."
```

## Build the Docker Image

To build the Docker image, run:

```
docker build -t stcr-user-study .
```

This will create an image tagged `stcr-user-study`.

## Run the Docker Container

To launch the app locally:

```
docker run -d -p 5000:5000 stcr-user-study
```

This runs the app in detached mode and maps port 5000 of the container to port 5000 on your host machine.

## Access the Website

Once running, open your browser and go to:

```
http://localhost:5000
```

## Saving and Retrieving Chat Logs

All chat transcripts and result logs are saved in:

```
/app/chat_folders/
```

To copy this data from the container to your machine:

1. Get the container ID:

```
docker ps
```

2. Use the `docker cp` command:

```
docker cp <container_id>:/app/chat_folders ./chat_folders_backup
```

This will create a `chat_folders_backup` directory with all saved interactions.

## Debugging and Troubleshooting

- Check if the container is running:

```
docker ps
```

- View container logs:

```
docker logs <container_id>
```

- Launch in interactive mode:

```
docker run -it -p 5000:5000 stcr-user-study
```

## Stopping the Container

To stop and remove the container:

```
docker stop <container_id>
docker rm <container_id>
```

## Notes

- The app will not work without a valid OpenAI API key.
- All chat sessions and study results are stored in `/app/chat_folders`.
- For more details on the algorithms and study setup, refer to the paper.