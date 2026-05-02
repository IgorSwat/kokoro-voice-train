from misaki import espeak
import importlib
import logging

logger = logging.getLogger("phonemizer")

class Phonemizer:
    def __init__(self, lang_code: str):
        """
        Initialize the Phonemizer with a specific language code.
        
        Args:
            lang_code: The language code (e.g., 'pl', 'de').
        """
        self.lang_code = lang_code
        self.g2p = espeak.EspeakG2P(language=lang_code)
        
        # Try to load language-specific fixup function
        self.fixup_fn = self._load_fixup(lang_code)

    def _load_fixup(self, lang_code: str):
        """Dynamically loads the fixup function for the given language."""
        module_name = f"tools.phonemes.fixup_{lang_code}"
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, "fixup"):
                return module.fixup
        except ImportError:
            pass
        return None

    def phonemize(self, text: str) -> str:
        """
        Convert text to phonemes and apply language-specific fixups.
        
        Args:
            text: Input text to phonemize.
            
        Returns:
            Phonemized and post-processed string.
        """
        phonemes, _ = self.g2p(text)
        
        # Language-specific post-processing
        if self.fixup_fn:
            phonemes = self.fixup_fn(phonemes)
            
        return phonemes
