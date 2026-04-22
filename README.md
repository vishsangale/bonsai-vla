# Bonsai-VLA

A from-scratch repository for understanding Vision-Language-Action models.

## Project Philosophy
1. **Clean Code**: Emphasize readability and teaching over hyper-optimization.
2. **From Scratch**: Rely on standard library or minimal PyTorch primitives where possible.
3. **Visual Learning**: Every complex concept should have an accompanying animation (using `manim`).

## Repository Structure
- `configs/`: Hydra configuration for experiments.
- `src/`: Core implementation of models and data loaders.
- `scripts/`: Entry point scripts for training and evaluation.
- `tests/`: Unit tests validating tensor math and shapes.
- `visualizations/`: Manim scripts for generating teaching animations.
 
## Infrastructure & Workflow
- **Queue Runner**: Long-running experiments (like 300-epoch training runs) are managed by the `mendu` queue system located at `../mendu/jobs/queue_runner.py`.
- **Job Configuration**: Add new jobs to `../mendu/jobs/queue.yaml` to have them run sequentially.

## Development Environment
- **Virtual Environment**: Always use the local `.venv` for running scripts and managing dependencies. See [AGENTS.md](file:///home/vishsangale/workspace/bonsai-vla/AGENTS.md) for detailed agent instructions.
- **Python Executable**: `.venv/bin/python3`
- **Activation**: `source .venv/bin/activate`
