# Import ConfigParser to read .ini configuration files
from configparser import ConfigParser
import os

class Config:
    """
    Config class is responsible for reading configuration values
    from the specified .ini file.
    """

    def __init__(self):
        """
        Constructor:
        - Creates a ConfigParser object
        - Reads the configuration file
        """
        # Get the directory where this Python file exists
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Build full path to ini file
        config_path = os.path.join(current_dir, "uiconfigfile.ini")

        self.config = ConfigParser()
        self.config.read(config_path)

    def get_page_title(self):
        """
        Returns the page title as a string.
        """
        return self.config["DEFAULT"].get("PAGE_TITLE")
    
    def get_llm_options(self):
        """
        Returns a list of LLM options.
        """
        return self.config["DEFAULT"].get("LLM_OPTIONS").split(", ")

    def get_usecase_options(self):
        """
        Returns a list of use case options.
        """
        return self.config["DEFAULT"].get("USECASE_OPTIONS").split(", ")

    def get_groq_model_options(self):
        """
        Returns a list of GROQ model options.
        """
        return self.config["DEFAULT"].get("GROQ_MODEL_OPTIONS").split(", ")

    
