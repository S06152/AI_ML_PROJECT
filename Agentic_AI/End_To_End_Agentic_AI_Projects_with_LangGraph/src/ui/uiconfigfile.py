# Import ConfigParser to read .ini configuration files
from configparser import ConfigParser
import os

class Config:
    """
    Config class is responsible for reading configuration values
    from the specified .ini file and providing helper methods
    to access those values safely.
    """

    def __init__(self, filename = "uiconfigfile.ini"):
        """
        Constructor:
        - Creates a ConfigParser object
        - Locates the configuration file
        - Reads the configuration file
        - Raises FileNotFoundError if file does not exist
        """
        # Get the directory where this Python file exists
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Build full path to the ini file
        self.config_path = os.path.join(current_dir, filename)

        # Create ConfigParser object
        self.config = ConfigParser()

        # Check if file exists before reading
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        # Read configuration file
        self.config.read(self.config_path)

    def get_page_title(self):
        """
        Returns:
            str: Page title defined in configuration.
        """
        return self.config["DEFAULT"].get("PAGE_TITLE")
    
    def get_llm_options(self):
        """
        Returns:
            list: List of available LLM options.
        """
        return self.get_list("LLM_OPTIONS")

    def get_usecase_options(self):
        """
        Returns:
            list: List of available use case options.
        """
        return self.get_list("USECASE_OPTIONS")

    def get_groq_model_options(self):
        """
        Returns:
            list: List of available GROQ model options.
        """
        return self.get_list("GROQ_MODEL_OPTIONS")

    def get_list(self, key):
        """
        Private helper method to safely fetch and split
        comma-separated configuration values.

        Args:
            key (str): Configuration key name.

        Returns:
            list: List of cleaned string values.
        """
        value = self.config["DEFAULT"].get(key, "")
        return [item.strip() for item in value.split(",") if item.strip()]



    
