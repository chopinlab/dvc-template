# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a minimal Python project called "dvc" that uses the modern `uv` package manager. The project currently contains a simple "Hello World" application.

## Development Commands

### Package Management
- `uv add <package>` - Add dependencies to the project
- `uv remove <package>` - Remove dependencies from the project
- `uv sync` - Update the project's environment
- `uv lock` - Update the project's lockfile

### Running Code
- `uv run main.py` - Run the main application
- `uv run python <script.py>` - Run any Python script in the project environment

### Environment Management
- `uv sync` - Ensure dependencies are installed and up to date

## Project Structure

- `main.py` - Entry point with a simple main function
- `pyproject.toml` - Project configuration and dependencies (uv/pip compatible)
- `uv.lock` - Dependency lockfile managed by uv

## Python Version

The project requires Python >= 3.13 as specified in pyproject.toml.