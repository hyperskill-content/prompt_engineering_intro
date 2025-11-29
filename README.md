# Introduction to prompt engineering

## Usage

To start the development environment:

### Create the .env file from a template
```shell
cp .env.template .env
```

### Build and start the container
```shell
docker compose up -d
```
Access Jupyter Lab at: http://localhost:8888

### Stop the container
```shell
docker compose down
```

The notebooks are mounted as a volume, so any changes you make in Jupyter will be reflected in your local `notebooks/` folder and vice versa.

If you want to add packages, edit `requirements.txt` and rebuild:
```shell
docker compose up -d --build
```