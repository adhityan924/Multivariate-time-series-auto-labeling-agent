import os
import numpy as np
from openai import OpenAI
from typing import Union, List
from colorama import Fore, Style

class EmbeddingUtils:
    def __init__(self, use_openai: bool = True):
        self.use_openai = use_openai
        if use_openai:
            self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            if not self.client.api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")

    def get_embedding(self, chunk: Union[List[float], np.ndarray]) -> np.ndarray:
        """Convert time series chunk to embedding vector"""
        if self.use_openai:
            print(Fore.YELLOW + "Using OpenAI embeddings..." + Style.RESET_ALL)
            return self._get_openai_embedding(chunk)
        else:
            print(Fore.YELLOW + "Using local embeddings..." + Style.RESET_ALL)
            return self._get_local_embedding(chunk)

    def _get_openai_embedding(self, chunk: Union[List[float], np.ndarray]) -> np.ndarray:
        """Get embedding from OpenAI API"""
        text_input = ", ".join(map(str, chunk))
        response = self.client.embeddings.create(
            input=text_input,
            model="text-embedding-ada-002"
        )
        return np.array(response.data[0].embedding)

    def _get_local_embedding(self, chunk: Union[List[float], np.ndarray]) -> np.ndarray:
        """Simple local embedding using normalized FFT"""
        fft = np.fft.fft(chunk)
        magnitude = np.abs(fft)[:len(chunk)//2]  # Take first half
        return magnitude / np.linalg.norm(magnitude)  # Normalize

# Singleton instance for use throughout the project
embedding_utils = EmbeddingUtils()
