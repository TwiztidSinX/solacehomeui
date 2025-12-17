"""
This module contains the shared global state of the application.
Importing this dictionary and modifying it will update the state for all modules.
"""

SYSTEM_STATE = {
    'current_model': None,
    'current_model_path_string': None,
    'current_backend': "llama.cpp"
}
