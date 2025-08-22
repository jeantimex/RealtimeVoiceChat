# Mac ARM64 Support Documentation

This document outlines the key changes made in the `mac-support` branch to ensure compatibility with Apple Silicon (ARM64) Macs. The primary motivation for these changes was to address issues with dependencies and hardware acceleration that are specific to the Mac ARM64 architecture.

## Summary of Changes

The following is a summary of the major changes:

1.  **Python Version:** The project now requires Python 3.11.
2.  **Coqui TTS Engine Disabled:** The Coqui TTS engine has been temporarily disabled due to an installation issue on Mac ARM64.
3.  **Wake Word Detection Engine:** Replaced `pvporcupine` with `openwakeword`.
4.  **PyTorch Device Selection:** Added support for Apple's Metal Performance Shaders (MPS).
5.  **Ollama Model Management:** Implemented automatic downloading of Ollama models.
6.  **Lazy Initialization:** Deferred the initialization of the audio recorder to prevent server startup delays.

---

### 1. Python 3.11 Requirement

*   **Why the change was made:**
    Many of the key dependencies, including PyTorch and other libraries with pre-compiled binaries for Apple Silicon, have better support and are more stable on Python 3.11. Standardizing on Python 3.11 ensures a more consistent and reliable experience for developers on Mac ARM64.

### 2. Coqui TTS Engine Disabled

*   **File Changed:** `code/server.py`

*   **Why the change was made:**
    The Coqui TTS engine is currently disabled in the `mac-support` branch because of a dependency issue on Mac ARM64. Specifically, the `deepspeed` library, which is a dependency of Coqui TTS, fails to install correctly. This is a known issue with `deepspeed` on Apple Silicon.

*   **How it was implemented:**
    The `TTS_START_ENGINE` variable in `code/server.py` has been commented out to prevent the Coqui TTS engine from being initialized.

### 3. Wake Word Detection: `openwakeword`

*   **Files Changed:**
    *   `code/transcribe.py`
    *   `code/wake_word_detector.py` (new file)

*   **Why the change was made:**
    The original wake word detection engine, `pvporcupine`, has limited support for Mac ARM64 and can be difficult to install and run reliably. To resolve this, we have replaced it with `openwakeword`, which is a more modern, open-source wake word engine with excellent support for Apple Silicon.

*   **How it was implemented:**
    A new `WakeWordDetector` class was created in `code/wake_word_detector.py` to encapsulate the `openwakeword` functionality. The `TranscriptionProcessor` in `code/transcribe.py` was then updated to use this new class, and the audio processing pipeline was modified to feed audio to the wake word detector before the speech-to-text engine.

### 4. PyTorch Device Selection for Apple Silicon

*   **File Changed:** `code/turndetect.py`

*   **Why the change was made:**
    PyTorch on Apple Silicon uses Metal Performance Shaders (MPS) for hardware acceleration, which is the equivalent of CUDA on NVIDIA GPUs. The original code only checked for CUDA or defaulted to the CPU.

*   **How it was implemented:**
    The device selection logic in `code/turndetect.py` was updated to explicitly check for the availability of `mps` and use it as the preferred device when CUDA is not available. This ensures that the turn detection model leverages the GPU on Apple Silicon Macs for better performance.

### 5. Automatic Ollama Model Downloading

*   **File Changed:** `code/llm_module.py`

*   **Why the change was made:**
    To improve the user experience, we've added a feature to automatically download Ollama models if they are not found locally. This removes the need for users to manually pull models before running the application.

*   **How it was implemented:**
    New functions were added to `code/llm_module.py` to check for the existence of an Ollama model and download it with a progress bar if it is missing. The `LLM` class now calls this functionality during initialization.

### 6. Lazy Initialization of the Audio Recorder

*   **File Changed:** `code/transcribe.py`

*   **Why the change was made:**
    Initializing the `AudioToTextRecorder` can take a few seconds, which could delay the startup of the web server. To prevent this, the recorder is now initialized "lazily."

*   **How it was implemented:**
    The recorder is no longer created in the `TranscriptionProcessor`'s constructor. Instead, it is initialized the first time audio is fed to the processor, ensuring the server can start up and respond to requests immediately.
