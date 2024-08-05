#  Copyright (c) 2024. Andrew Florjancic

"""
Basic module to display loading state information to the user
"""

def loading(message: str):
    """
    Prints the provided message with a space following the message and no newline
    :param message: The loading message that will be printed to the console
    """
    print(message, end=" ")


def done():
    """Prints a green checkmark indicating the current step is complete"""
    print('\u2705')