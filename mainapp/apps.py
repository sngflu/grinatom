from django.apps import AppConfig


class MainappConfig(AppConfig):
    # Specify the default field type for auto-generated fields
    default_auto_field = 'django.db.models.BigAutoField'
    
    # Define the name of the app
    name = 'mainapp'

    # This method is called when the application is ready to be used
    def ready(self):
        # Import the function to load models and tokenizers
        from .utils.preprocessing import load_model_and_tokenizer
        
        # Call the function to load models and tokenizers when the app is ready
        load_model_and_tokenizer()
