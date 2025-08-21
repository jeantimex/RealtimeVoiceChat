import logging
import numpy as np
import threading
import queue
import time
from typing import List, Optional, Callable

logger = logging.getLogger(__name__)

# Try to import OpenWakeWord for wake word detection
try:
    from openwakeword import Model as WakeWordModel
    OPENWAKEWORD_AVAILABLE = True
    logger.info("👂🔊 OpenWakeWord available for wake word detection")
except ImportError:
    OPENWAKEWORD_AVAILABLE = False
    logger.warning("👂⚠️ OpenWakeWord not available. Wake word detection disabled.")
    WakeWordModel = None


class WakeWordDetector:
    """
    Custom wake word detection using OpenWakeWord.
    
    This class runs independently of RealtimeSTT and provides wake word detection
    using the OpenWakeWord library, which has proper Apple Silicon support.
    """
    
    def __init__(
        self, 
        wake_words: List[str] = None,
        threshold: float = 0.5,
        on_wake_word_detected: Optional[Callable[[str], None]] = None
    ):
        """
        Initialize the wake word detector.
        
        Args:
            wake_words: List of wake words to detect (e.g., ["jarvis", "hey_jarvis"])
            threshold: Detection threshold (0.0 to 1.0)
            on_wake_word_detected: Callback when wake word is detected
        """
        self.wake_words = wake_words or ["jarvis"]
        self.threshold = threshold
        self.on_wake_word_detected = on_wake_word_detected
        self.model = None
        self.audio_queue = queue.Queue()
        self.running = False
        self.detection_thread = None
        
        if OPENWAKEWORD_AVAILABLE:
            self._initialize_model()
        else:
            logger.error("👂💥 Cannot initialize wake word detector: OpenWakeWord not available")
    
    def _initialize_model(self):
        """Initialize the OpenWakeWord model."""
        try:
            # Initialize with default models (includes common wake words)
            self.model = WakeWordModel()
            logger.info(f"👂🔊 Wake word detector initialized for: {self.wake_words}")
            logger.info(f"👂🔊 Available models: {list(self.model.models.keys())}")
        except Exception as e:
            logger.error(f"👂💥 Failed to initialize wake word model: {e}")
            self.model = None
    
    def start(self):
        """Start the wake word detection in a background thread."""
        if not self.model:
            logger.warning("👂⚠️ Cannot start wake word detection: Model not initialized")
            return
            
        self.running = True
        self.detection_thread = threading.Thread(target=self._detection_worker, daemon=True)
        self.detection_thread.start()
        logger.info("👂🚀 Wake word detection started")
    
    def stop(self):
        """Stop the wake word detection."""
        self.running = False
        if self.detection_thread:
            self.detection_thread.join(timeout=1.0)
        logger.info("👂🛑 Wake word detection stopped")
    
    def feed_audio(self, audio_chunk: np.ndarray):
        """
        Feed audio data to the wake word detector.
        
        Args:
            audio_chunk: Audio data as float32 numpy array, normalized to [-1.0, 1.0]
        """
        if self.running and not self.audio_queue.full():
            try:
                self.audio_queue.put_nowait(audio_chunk)
            except queue.Full:
                pass  # Drop audio if queue is full
    
    def _detection_worker(self):
        """Background thread worker for wake word detection."""
        logger.info("👂🔍 Wake word detection worker started")
        
        while self.running:
            try:
                # Get audio chunk with timeout
                audio_chunk = self.audio_queue.get(timeout=0.1)
                
                # Convert to the format expected by OpenWakeWord (16kHz, int16)
                if audio_chunk.dtype != np.float32:
                    audio_chunk = audio_chunk.astype(np.float32)
                
                # OpenWakeWord expects int16 audio at 16kHz
                audio_int16 = (audio_chunk * 32767).astype(np.int16)
                
                # Run detection
                prediction = self.model.predict(audio_int16)
                
                # Check if any wake word was detected above threshold
                for wake_word in self.wake_words:
                    # Try different variations of the wake word name
                    possible_names = [
                        wake_word,
                        wake_word.lower(),
                        wake_word.replace(" ", "_"),
                        f"hey_{wake_word.lower()}"
                    ]
                    
                    for name in possible_names:
                        if name in prediction and prediction[name] > self.threshold:
                            logger.info(f"👂🎯 Wake word detected: '{name}' (confidence: {prediction[name]:.3f})")
                            if self.on_wake_word_detected:
                                self.on_wake_word_detected(wake_word)
                            break
                    else:
                        continue
                    break  # Break outer loop if wake word found
                        
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"👂💥 Error in wake word detection: {e}")
                time.sleep(0.1)
        
        logger.info("👂🔍 Wake word detection worker stopped")