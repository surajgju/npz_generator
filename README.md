# NPZ Generator

A standalone tool for generating NPZ files from audio and visualizing motion.

## Usage

You can generate motion data and visualizations by simply running:

```bash
python3 generate_npz.py
```

### Features & Defaults

- **Default Folders**: The script now defaults to reading from `./input` and saving to `./output`, so you no longer need to pass these arguments manually.
- **Default Visualization**: Visualization is now enabled by default. If you need to disable it, you can use the `--no_visualization` flag:
    ```bash
    python3 generate_npz.py --no_visualization
    ```
- **Improved Audio Support**: The script supports various audio formats beyond just `.wav`, including `.mp3`, `.m4a`, `.flac`, and `.ogg`. It will automatically detect files like `intro.mp3` in your `./input` directory.

## Configuration

If you still need to customize paths, you can use the following arguments:

- `--audio_folder`: Path to the folder containing audio files (default: `./input`).
- `--save_folder`: Path to the folder where NPZ and MP4 files will be saved (default: `./output`).
- `--no_visualization`: Add this flag to skip the visualization step.
- `--model_folder`: Path to the SMPL-X models (default: `./emage_evaltools/smplx_models/`).
