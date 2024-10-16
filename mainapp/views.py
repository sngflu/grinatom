from django.shortcuts import render
from .utils.preprocessing import predict_sentiment, predict_score

# View for the index page
def index(request):
    # Initialize an empty context
    context = {}
    
    # Check if the request method is POST
    if request.method == "POST":
        # Get the input text from the form
        input_text = request.POST.get("input_text", "")
        
        # If input text is provided, proceed with prediction
        if input_text:
            try:
                # Predict the sentiment (1 for positive, 0 for negative)
                sentiment = predict_sentiment(input_text)
                
                # Set sentiment text based on prediction
                if sentiment == 1:
                    sentiment_text = "Позитивный комментарий"
                else:
                    sentiment_text = "Негативный комментарий"
                
                # Predict the score based on the sentiment and input text
                score = predict_score(input_text, sentiment)
                
                # Update the context with input text, sentiment, and score
                context = {
                    "input_text": input_text,
                    "sentiment": sentiment_text,
                    "score": score
                }
            # Handle any exceptions and display an error message
            except Exception as e:
                context = {
                    "input_text": input_text,
                    "sentiment": f"Ошибка при обработке: {str(e)}"
                }
    
    # Render the index.html template with the context
    return render(request, "index.html", context)
