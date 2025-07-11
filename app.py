from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import sys
import os

# Add current directory to path to import the recommendation module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the recommendation system
try:
    from movie_recommendation_system import get_recommendations, df
except ImportError as e:
    print(f"Error importing recommendation system: {e}")
    sys.exit(1)

app = Flask(__name__)

# Configure CORS to allow requests from any origin
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "Access-Control-Allow-Credentials"],
        "supports_credentials": True
    }
})


@app.route('/')
def index():
    """Serve the main page with search form"""
    # Get some sample movies for suggestions
    sample_movies = df['Series_Title'].head(10).tolist()
    total_movies = len(df)

    # Check if there's a search query from 404 redirect
    search_query = request.args.get('search', '')

    return render_template(
        'index.html',
        sample_movies=sample_movies,
        total_movies=total_movies,
        search_query=search_query
    )


@app.route('/recommend', methods=['POST'])
def recommend_movies():
    """
    POST endpoint to get movie recommendations
    Expects JSON: {"movie_name": "movie title"}
    Returns JSON: {"recommendations": [...], "movie_title": "...", ...}
    """
    try:
        # Get JSON data from request
        data = request.get_json()

        if not data or 'movie_name' not in data:
            return jsonify({
                'error': 'Missing movie_name in request body',
                'usage': 'POST /recommend with JSON body: {"movie_name": "movie title"}'
            }), 400

        movie_name = data['movie_name'].strip()

        if not movie_name:
            return jsonify({'error': 'Movie name cannot be empty'}), 400

        # Get recommendations using the imported function
        recommendations = get_recommendations(movie_name)

        # Convert to list if it's a pandas Series
        recommendations_list = list(recommendations) if hasattr(
            recommendations, '__iter__') else []

        # Check if we used a closest match
        exact_match = any(df['Series_Title'].str.lower() == movie_name.lower())
        used_closest_match = not exact_match and len(recommendations_list) > 0
        closest_match = None

        if used_closest_match:
            from difflib import get_close_matches
            possible_titles = df['Series_Title'].tolist()
            close_matches = get_close_matches(
                movie_name, possible_titles, n=1, cutoff=0.6)
            closest_match = close_matches[0] if close_matches else None

        response_data = {
            'success': True,
            'movie_title': movie_name,
            'recommendations': recommendations_list,
            'total_recommendations': len(recommendations_list),
            'used_closest_match': used_closest_match,
            'closest_match': closest_match
        }

        return jsonify(response_data), 200

    except Exception as e:
        return jsonify({
            'error': f'Internal server error: {str(e)}',
            'success': False
        }), 500


@app.route('/api/movies', methods=['GET'])
def get_all_movies():
    """Get all available movies in the database"""
    try:
        movies = df['Series_Title'].tolist()
        return jsonify({
            'success': True,
            'movies': movies,
            'total_count': len(movies)
        }), 200
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/search', methods=['GET'])
def search_movies():
    """Search for movies by partial name match"""
    try:
        query = request.args.get('q', '').strip()

        if not query:
            return jsonify({'error': 'Query parameter "q" is required'}), 400

        # Find movies that contain the query string
        matching_movies = df[df['Series_Title'].str.contains(
            query, case=False, na=False)]
        matches = matching_movies['Series_Title'].tolist()

        return jsonify({
            'success': True,
            'query': query,
            'matches': matches,
            'count': len(matches)
        }), 200

    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Movie Recommendation API',
        'dataset_loaded': len(df) > 0,
        'total_movies': len(df)
    }), 200


# ============ ERROR HANDLERS ============

@app.errorhandler(404)
def not_found(error):
    """Custom 404 error page"""
    return render_template('404.html'), 404


@app.errorhandler(405)
def method_not_allowed(error):
    """Handle method not allowed errors"""
    return jsonify({
        'error': 'Method not allowed',
        'message': 'The method is not allowed for the requested URL.',
        'allowed_methods': ['GET', 'POST']
    }), 405


@app.errorhandler(500)
def internal_error(error):
    """Handle internal server errors"""
    return jsonify({
        'error': 'Internal server error',
        'message': 'Something went wrong on our end. Please try again later.'
    }), 500


@app.errorhandler(400)
def bad_request(error):
    """Handle bad request errors"""
    return jsonify({
        'error': 'Bad request',
        'message': 'The request could not be understood by the server.'
    }), 400


@app.errorhandler(403)
def forbidden(error):
    """Handle forbidden errors"""
    return jsonify({
        'error': 'Forbidden',
        'message': 'You do not have permission to access this resource.'
    }), 403


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
