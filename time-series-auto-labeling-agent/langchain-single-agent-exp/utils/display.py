"""
Utility functions for displaying colorized output.
"""
import colorama
from colorama import Fore, Style

# Initialize colorama
colorama.init()

def print_step(message: str, *args) -> None:
    """
    Print a step message in cyan.
    
    Args:
        message: Message to print.
        *args: Additional arguments to print.
    """
    print(f"{Fore.CYAN}{Style.BRIGHT}[STEP] {message}{Style.RESET_ALL}", *args)

def print_info(message: str, *args) -> None:
    """
    Print an info message in blue.
    
    Args:
        message: Message to print.
        *args: Additional arguments to print.
    """
    print(f"{Fore.BLUE}[INFO] {message}{Style.RESET_ALL}", *args)

def print_success(message: str, *args) -> None:
    """
    Print a success message in green.
    
    Args:
        message: Message to print.
        *args: Additional arguments to print.
    """
    print(f"{Fore.GREEN}{Style.BRIGHT}[SUCCESS] {message}{Style.RESET_ALL}", *args)

def print_warning(message: str, *args) -> None:
    """
    Print a warning message in yellow.
    
    Args:
        message: Message to print.
        *args: Additional arguments to print.
    """
    print(f"{Fore.YELLOW}[WARNING] {message}{Style.RESET_ALL}", *args)

def print_error(message: str, *args) -> None:
    """
    Print an error message in red.
    
    Args:
        message: Message to print.
        *args: Additional arguments to print.
    """
    print(f"{Fore.RED}{Style.BRIGHT}[ERROR] {message}{Style.RESET_ALL}", *args) 