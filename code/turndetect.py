import logging
logger = logging.getLogger(__name__)

import transformers
import collections
import threading
import queue
import torch
import time
import re

# Configuration constants
model_dir_local = "KoljaB/SentenceFinishedClassification"
model_dir_cloud = "/root/models/sentenceclassification/"
sentence_end_marks = ['.', '!', '?', '。'] # Characters considered sentence endings

# Anchor points for probability-to-pause interpolation
anchor_points = [
    (0.0, 1.0), # Probability 0.0 maps to pause 1.0
    (1.0, 0.0)  # Probability 1.0 maps to pause 0.0
]

def ends_with_string(text: str, s: str) -> bool:
    """
    Checks if a string ends with a specific substring, allowing for one trailing character.

    Useful for checking punctuation even if there's a trailing space.

    Args:
        text: The input string to check.
        s: The substring to look for at the end.

    Returns:
        True if the text ends with s, or if text[:-1] ends with s. False otherwise.
    """
    if text.endswith(s):
        return True
    # Check if it ends if we ignore the very last character (e.g., trailing space)
    if len(text) > 1 and text[:-1].endswith(s):
        return True
    return False

def preprocess_text(text: str) -> str:
    """
    Cleans and normalizes the beginning of a text string.

    Applies the following steps:
    1. Removes leading whitespace.
    2. Removes leading ellipses ("...") if present.
    3. Removes leading whitespace again (after potential ellipses removal).
    4. Uppercases the first letter of the remaining text.

    Args:
        text: The input text string.

    Returns:
        The preprocessed text string.
    """
    text = text.lstrip() # Remove leading whitespaces
    if text.startswith("..."): #  Remove starting ellipses if present
        text = text[3:]
    text = text.lstrip() # Remove any leading whitespaces again after ellipses removal
    if text:
        text = text[0].upper() + text[1:] # Uppercase the first letter

    return text

def strip_ending_punctuation(text: str) -> str:
    """
    Removes trailing punctuation marks defined in `sentence_end_marks`.

    Removes trailing whitespace first, then iteratively removes any characters
    from `sentence_end_marks` found at the end of the string.

    Args:
        text: The input text string.

    Returns:
        The text string with specified trailing punctuation removed.
    """
    text = text.rstrip()
    for char in sentence_end_marks:
        # Repeatedly strip each punctuation mark in case of multiples (e.g., "!!")
        while text.endswith(char):
             text = text.rstrip(char)
    return text # Return the stripped text

def find_matching_texts(texts_without_punctuation: collections.deque) -> list[tuple[str, str]]:
    """
    Finds recent consecutive entries with the same stripped text.

    Iterates backwards through the deque of (original_text, stripped_text) tuples.
    It collects all entries matching the stripped text of the *last* entry,
    stopping as soon as a non-matching stripped text is encountered.

    Args:
        texts_without_punctuation: A deque of tuples, where each tuple is
                                   (original_text, stripped_text).

    Returns:
        A list of matching (original_text, stripped_text) tuples in their
        original order (oldest matching first). Returns an empty list if the
        input deque is empty.
    """
    if not texts_without_punctuation:
        return []

    # Get the stripped text from the last entry
    last_stripped_text = texts_without_punctuation[-1][1]

    matching_entries = []

    # Iterate through the deque backwards
    for entry in reversed(texts_without_punctuation):
        original_text, stripped_text = entry

        # If we find a non-match, stop collecting
        if stripped_text != last_stripped_text:
            break

        # Add the matching entry (will be reversed later)
        matching_entries.append(entry)

    # Reverse the results to maintain original insertion order
    matching_entries.reverse()

    return matching_entries

def interpolate_detection(prob: float) -> float:
    """
    Linearly interpolates a value based on probability using predefined anchor points.

    Maps an input probability `prob` (clamped between 0.0 and 1.0) to an output
    value based on the `anchor_points` list. It finds the segment within
    `anchor_points` where `prob` falls and performs linear interpolation.

    Args:
        prob: The input probability, expected between 0.0 and 1.0.

    Returns:
        The interpolated value. Returns 4.0 as a fallback if interpolation fails
        (though this shouldn't happen if anchor points cover the [0,1] range).
    """
    # Clamp probability between 0.0 and 1.0 just in case
    p = max(0.0, min(prob, 1.0))

    # Check if probability exactly matches an anchor point
    for ap_p, ap_val in anchor_points:
        if abs(ap_p - p) < 1e-9: # Use tolerance for float comparison
            return ap_val

    # Find the segment [p1, p2] where p resides
    for i in range(len(anchor_points) - 1):
        p1, v1 = anchor_points[i]
        p2, v2 = anchor_points[i+1]
        # Ensure anchor points are sorted by probability for this logic to work
        if p1 <= p <= p2:
            # Avoid division by zero if anchor points have the same probability
            if abs(p2 - p1) < 1e-9:
                return v1 # or v2, they should be close if p1=p2
            # Linear interpolation formula
            ratio = (p - p1) / (p2 - p1)
            return v1 + ratio * (v2 - v1)

    # Fallback: Should not be reached if anchor_points cover [0,1] properly.
    logger.warning(f"🎤⚠️ Probability {p} fell outside defined anchor points {anchor_points}. Returning fallback value.")
    return 4.0

class TurnDetection:
    """
    Manages turn detection logic based on text input and sentence completion model.

    This class receives text segments, uses a transformer model to predict sentence
    completion probability, considers punctuation, and calculates a suggested waiting
    time (pause duration) before the next speaker might start. It uses a background
    thread for processing and provides a callback for new waiting time suggestions.
    It also maintains a history of recent texts and uses caching for model predictions.
    """

    def __init__(
        self,
        on_new_waiting_time: callable,
        local: bool = False,
        pipeline_latency: float = 0.5,
        pipeline_latency_overhead: float = 0.1,
    ) -> None:
        """
        Initializes the TurnDetection instance.

        Loads the sentence classification model and tokenizer, sets up internal state
        (deques, cache), starts the background processing thread, and performs model warmup.

        Args:
            on_new_waiting_time: Callback function invoked when a new waiting time is calculated.
                                 It receives `(time: float, text: str)`.
            local: If True, loads the model from `model_dir_local`, otherwise from `model_dir_cloud`.
            pipeline_latency: Estimated base latency of the STT/processing pipeline in seconds.
            pipeline_latency_overhead: Additional buffer added to the pipeline latency.
        """
        model_dir = model_dir_local if local else model_dir_cloud

        self.on_new_waiting_time = on_new_waiting_time

        self.current_waiting_time: float = -1 # Tracks the last suggested time
        # Use deques with maxlen for efficient, bounded history storage
        self.text_time_deque: collections.deque[tuple[float, str]] = collections.deque(maxlen=100)
        self.texts_without_punctuation: collections.deque[tuple[str, str]] = collections.deque(maxlen=20)

        self.text_queue: queue.Queue[str] = queue.Queue() # Queue for incoming text
        self.text_worker = threading.Thread(
            target=self._text_worker,
            daemon=True # Allows program to exit even if this thread is running
        )
        self.text_worker.start()

        # Prefer CUDA > MPS (Apple Silicon) > CPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        logger.info(f"🎤🔌 Using device: {self.device}")
        self.tokenizer = transformers.DistilBertTokenizerFast.from_pretrained(model_dir)
        self.classification_model = transformers.DistilBertForSequenceClassification.from_pretrained(model_dir)
        self.classification_model.to(self.device)
        self.classification_model.eval() # Set model to evaluation mode
        self.max_length: int = 128 # Max sequence length for the model
        self.pipeline_latency: float = pipeline_latency
        self.pipeline_latency_overhead: float = pipeline_latency_overhead

        # Initialize completion probability cache with OrderedDict for LRU behavior
        self._completion_probability_cache: collections.OrderedDict[str, float] = collections.OrderedDict()
        self._completion_probability_cache_max_size: int = 256 # Max size for the LRU cache

        # Warmup the classification model for faster initial predictions
        logger.info("🎤🔥 Warming up the classification model...")
        with torch.no_grad():
            warmup_text = "This is a warmup sentence."
            inputs = self.tokenizer(
                warmup_text,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=self.max_length
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            _ = self.classification_model(**inputs) # Run one prediction
        logger.info("🎤✅ Classification model warmed up.")

        # Default dynamic pause settings (initialized for speed_factor=0.0)
        self.detection_speed: float = 0.5
        self.ellipsis_pause: float = 2.3
        self.punctuation_pause: float = 0.39
        self.exclamation_pause: float = 0.35
        self.question_pause: float = 0.33
        self.unknown_sentence_detection_pause: float = 1.25
        # Apply initial settings (can be called again later)
        self.update_settings(speed_factor=0.0)

    def update_settings(self, speed_factor: float) -> None:
        """
        Adjusts dynamic pause parameters based on a speed factor.

        Linearly interpolates between 'fast' (speed_factor=0.0) and 'very_slow'
        (speed_factor=1.0) settings for various pause durations used in calculation.
        Clamps speed_factor between 0.0 and 1.0.

        Args:
            speed_factor: A float between 0.0 (fastest) and 1.0 (slowest) controlling
                          the interpolation between predefined settings.
        """
        speed_factor = max(0.0, min(speed_factor, 1.0)) # Clamp factor

        # Base 'fast' settings (speed_factor = 0.0)
        fast = {
            'detection_speed': 0.5,
            'ellipsis_pause': 2.3,
            'punctuation_pause': 0.39,
            'exclamation_pause': 0.35,
            'question_pause': 0.33,
            'unknown_sentence_detection_pause': 1.25
        }

        # Target 'very slow' settings (speed_factor = 1.0)
        very_slow = {
            'detection_speed': 1.7,
            'ellipsis_pause': 3.0,
            'punctuation_pause': 0.9,
            'exclamation_pause': 0.8,
            'question_pause': 0.8,
            'unknown_sentence_detection_pause': 1.9
        }

        # Linear interpolation for each parameter
        self.detection_speed = fast['detection_speed'] + speed_factor * (very_slow['detection_speed'] - fast['detection_speed'])
        self.ellipsis_pause = fast['ellipsis_pause'] + speed_factor * (very_slow['ellipsis_pause'] - fast['ellipsis_pause'])
        self.punctuation_pause = fast['punctuation_pause'] + speed_factor * (very_slow['punctuation_pause'] - fast['punctuation_pause'])
        self.exclamation_pause = fast['exclamation_pause'] + speed_factor * (very_slow['exclamation_pause'] - fast['exclamation_pause'])
        self.question_pause = fast['question_pause'] + speed_factor * (very_slow['question_pause'] - fast['question_pause'])
        self.unknown_sentence_detection_pause = fast['unknown_sentence_detection_pause'] + speed_factor * (very_slow['unknown_sentence_detection_pause'] - fast['unknown_sentence_detection_pause'])
        logger.info(f"🎤⚙️ Updated turn detection settings with speed_factor={speed_factor:.2f}")


    def suggest_time(
            self,
            time_val: float, # Renamed from 'time' to avoid shadowing builtin
            text: str = None
        ) -> None:
        """
        Invokes the `on_new_waiting_time` callback with the suggested pause duration.

        Only triggers the callback if the new suggested time is different from the
        currently stored `current_waiting_time` to avoid redundant calls.

        Args:
            time_val: The calculated suggested waiting time in seconds.
            text: The text segment associated with this waiting time calculation.
        """
        if time_val == self.current_waiting_time:
            return # No change, do nothing

        self.current_waiting_time = time_val

        if self.on_new_waiting_time:
            self.on_new_waiting_time(time_val, text)

    def get_completion_probability(
        self,
        sentence: str
    ) -> float:
        """
        Calculates the probability that the given sentence is complete using the ML model.

        Uses an internal LRU cache (`_completion_probability_cache`) to store and
        retrieve results for previously seen sentences, improving performance.

        Args:
            sentence: The input sentence string to analyze.

        Returns:
            A float representing the probability (between 0.0 and 1.0) that the
            sentence is considered complete by the model.
        """
        # Check cache first
        if sentence in self._completion_probability_cache:
            self._completion_probability_cache.move_to_end(sentence) # Mark as recently used
            return self._completion_probability_cache[sentence]

        # If not in cache, run model prediction
        import torch
        import torch.nn.functional as F

        inputs = self.tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_length
        )
        # Move input tensors to the correct device (CPU or GPU)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad(): # Disable gradient calculation for inference
            outputs = self.classification_model(**inputs)

        logits = outputs.logits
        # Apply softmax to get probabilities [prob_incomplete, prob_complete]
        probabilities = F.softmax(logits, dim=1).squeeze().tolist()
        prob_complete = probabilities[1] # Index 1 corresponds to 'complete' label

        # Store the result in the cache
        self._completion_probability_cache[sentence] = prob_complete
        self._completion_probability_cache.move_to_end(sentence) # Mark as recently used

        # Maintain cache size (LRU eviction)
        if len(self._completion_probability_cache) > self._completion_probability_cache_max_size:
            self._completion_probability_cache.popitem(last=False) # Remove the least recently used item

        return prob_complete

    def get_suggested_whisper_pause(self, text: str) -> float:
        """
        Determines a base pause duration based on the text's ending punctuation.

        Checks for specific ending patterns ('...', '.', '!', '?') and returns
        a corresponding pause duration defined by the instance's settings
        (e.g., `self.ellipsis_pause`). Returns `self.unknown_sentence_detection_pause`
        if no specific punctuation is matched.

        Args:
            text: The input text string.

        Returns:
            The suggested pause duration in seconds based on ending punctuation.
        """
        if ends_with_string(text, "..."):
            return self.ellipsis_pause
        elif ends_with_string(text, "."):
            return self.punctuation_pause
        elif ends_with_string(text, "!"):
            return self.exclamation_pause
        elif ends_with_string(text, "?"):
            return self.question_pause
        else:
            # No specific ending detected, use the general pause for unknown endings
            return self.unknown_sentence_detection_pause

    def _text_worker(
        self
    ) -> None:
        """
        Background worker thread that processes text from the queue for turn detection.

        Continuously retrieves text items from `self.text_queue`. For each item, it:
        1. Preprocesses the text.
        2. Updates text history deques.
        3. Finds recent matching text segments to analyze punctuation consistency.
        4. Calculates an average pause based on observed punctuation in matches.
        5. Cleans the text further for the sentence completion model.
        6. Gets the completion probability from the model (using cache).
        7. Interpolates the model's probability to another pause value.
        8. Combines the punctuation-based pause and model-based pause using weighting.
        9. Applies a speed factor and adjustments (e.g., for ellipses).
        10. Ensures the final pause meets minimum pipeline latency requirements.
        11. Calls `suggest_time` with the final calculated pause duration.
        Handles queue timeouts gracefully to allow for potential shutdown.
        """
        while True:
            try:
                # Wait for text from the queue, with a timeout to avoid blocking forever
                text = self.text_queue.get(block=True, timeout=0.1)
            except queue.Empty:
                # No text received within the timeout, loop again
                time.sleep(0.01) # Small sleep to yield CPU when idle
                continue

            # --- Processing starts when text is received ---
            logger.info(f"🎤⚙️ Starting pause calculation for: \"{text}\"")
            
            processed_text = preprocess_text(text) # Apply initial cleaning

            # Update history deques
            current_time = time.time()
            self.text_time_deque.append((current_time, processed_text))
            text_without_punctuation = strip_ending_punctuation(processed_text)
            self.texts_without_punctuation.append((processed_text, text_without_punctuation))

            # Analyze recent matching texts for consistent punctuation pauses
            matches = find_matching_texts(self.texts_without_punctuation)

            added_pauses = 0
            contains_ellipses = False
            if matches: # Avoid division by zero if matches is empty
                for i, match in enumerate(matches):
                    same_text, _ = match # We only need the original text here
                    whisper_suggested_pause_match = self.get_suggested_whisper_pause(same_text)
                    added_pauses += whisper_suggested_pause_match
                    if ends_with_string(same_text, "..."):
                        contains_ellipses = True
                # Calculate average pause based on recent consistent segments
                avg_pause = added_pauses / len(matches)
            else:
                # If no matches, use the pause suggested by the current text directly
                 avg_pause = self.get_suggested_whisper_pause(processed_text)
                 if ends_with_string(processed_text, "..."):
                    contains_ellipses = True

            whisper_suggested_pause = avg_pause # Use the averaged pause

            # Prepare text for the sentence completion model (remove all punctuation)
            import string
            transtext = processed_text.translate(str.maketrans('', '', string.punctuation))
            # Further clean potentially remaining non-alphanumeric chars at the end
            cleaned_for_model = re.sub(r'[^a-zA-Z\s]+$', '', transtext).rstrip() # Also remove trailing spaces

            # Get sentence completion probability
            prob_complete = self.get_completion_probability(cleaned_for_model)

            # Interpolate probability to a pause duration
            sentence_finished_model_pause = interpolate_detection(prob_complete)

            # Combine pauses: weighted average giving more importance to punctuation pause
            weight_towards_whisper = 0.65
            weighted_pause = (weight_towards_whisper * whisper_suggested_pause +
                             (1 - weight_towards_whisper) * sentence_finished_model_pause)

            # Apply overall speed factor
            final_pause = weighted_pause * self.detection_speed

            # Add slight extra pause if ellipses were detected recently
            if contains_ellipses:
                final_pause += 0.2

            logger.info(f"🎤📊 Calculated pauses: Punct={whisper_suggested_pause:.2f}, Model={sentence_finished_model_pause:.2f}, Weighted={weighted_pause:.2f}, Final={final_pause:.2f} for \"{processed_text}\" (Prob={prob_complete:.2f})")


            # Ensure final pause is not less than the pipeline latency overhead
            min_pause = self.pipeline_latency + self.pipeline_latency_overhead
            if final_pause < min_pause:
                logger.info(f"🎤⚠️ Final pause ({final_pause:.2f}s) is less than minimum ({min_pause:.2f}s). Using minimum.")
                final_pause = min_pause
            
            # Suggest the calculated time via callback
            self.suggest_time(final_pause, processed_text) # Use processed_text for context

            # Mark task as done for the queue (important if using queue.join())
            self.text_queue.task_done()

    def calculate_waiting_time(
            self,
            text: str) -> None:
        """
        Adds a text segment to the processing queue for waiting time calculation.

        This is the entry point for feeding text into the turn detection system.
        The actual calculation happens asynchronously in the background worker thread.

        Args:
            text: The text segment (e.g., from STT) to be processed.
        """
        logger.info(f"🎤📥 Queuing text for pause calculation: \"{text}\"")
        self.text_queue.put(text)

    def reset(self) -> None:
        """
        Resets the internal state of the TurnDetection instance.

        Clears the text history deques, the model prediction cache, and resets the
        current waiting time tracker. Useful for starting a new conversation or
        interaction context.
        """
        logger.info("🎤🔄 Resetting TurnDetection state.")
        # Clear the history deques
        self.text_time_deque.clear()
        self.texts_without_punctuation.clear()
        # Reset the last suggested time
        self.current_waiting_time = -1
        # Clear the prediction cache
        if hasattr(self, "_completion_probability_cache"):
            self._completion_probability_cache.clear()
        # Clear the processing queue (optional, might discard unprocessed items)
        # while not self.text_queue.empty():
        #     try:
        #         self.text_queue.get_nowait()
        #         self.text_queue.task_done()
        #     except queue.Empty:
        #         break