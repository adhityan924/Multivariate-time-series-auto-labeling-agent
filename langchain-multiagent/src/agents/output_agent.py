import json
from colorama import Fore, Style
from typing import List, Dict, Any

# No Langchain agent boilerplate here, as it's purely formatting.
# A simple function is sufficient and more efficient.

def format_and_print_results(annotation_label: str, similar_chunks: List[List[float]], top_k: int):
    """
    Formats the similarity search results into the required dictionary
    structure and prints it to the console in a pretty JSON format.

    Args:
        annotation_label: The label provided by the user.
        similar_chunks: A list of the top-k similar chunks found (each chunk is a list of floats).
        top_k: The number of results requested (for context in messages).
    """
    print(f"\n{Fore.MAGENTA}--- Search Results ---{Style.RESET_ALL}")

    if not similar_chunks:
        print(f"{Fore.YELLOW}No similar chunks found for the provided query.{Style.RESET_ALL}")
        output_dict = {annotation_label: []}
    else:
        # Ensure we only take up to top_k results, in case the search returned more/less somehow
        output_dict = {
            annotation_label: similar_chunks[:top_k]
        }
        print(f"{Fore.GREEN}Top {len(output_dict[annotation_label])} similar chunks found for label '{annotation_label}':{Style.RESET_ALL}")

    # Pretty print the JSON output
    try:
        pretty_json = json.dumps(output_dict, indent=2)
        print(f"{Fore.WHITE}{pretty_json}{Style.RESET_ALL}")
    except TypeError as e:
        print(f"{Fore.RED}Error formatting results to JSON: {e}{Style.RESET_ALL}")
        # Print the raw dictionary as a fallback
        print(output_dict)

    print(f"{Fore.MAGENTA}----------------------{Style.RESET_ALL}")
