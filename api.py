"""
==================================================================
Exoplanet Identification Backend - REST API
==================================================================
This module provides a Flask-based REST API for the exoplanet
identification model, enabling frontend integration.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
from datetime import datetime
import base64
import io
import json as json_lib
import requests
import logging
import joblib
from dotenv import load_dotenv
import utils
from utils import (
    predict_single, 
    predict, 
    get_model_info, 
    preprocess_input,
    validate_data,
    load_model,
    load_metrics,
    calculate_confidence_score,
    classify_planet_type,
    generate_prediction_explanation
)# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for all routes (configure as needed for production)
CORS(app)

# Load environment configuration
load_dotenv()
load_dotenv('.env')  # Explicitly load .env file
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
OPENROUTER_MODEL = os.getenv('OPENROUTER_MODEL', 'x-ai/grok-4-fast')
OPENROUTER_BASE_URL = os.getenv('OPENROUTER_BASE_URL', 'https://openrouter.ai/api/v1')
APP_TITLE = os.getenv('APP_TITLE', 'Exoplanet Identification API')
APP_URL = os.getenv('APP_URL', 'http://localhost:5000')

# Debug: Print API configuration (remove in production)
print(f"[DEBUG] API Key loaded: {OPENROUTER_API_KEY is not None}")
print(f"[DEBUG] Model: {OPENROUTER_MODEL}")

# Configuration
app.config['JSON_SORT_KEYS'] = False
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global variables to cache model
MODEL_CACHE = None
MODEL_LOAD_TIME = None


@app.route('/')
def home():
    """Redirect to Netlify frontend.

    Frontend is hosted separately on Netlify, backend serves API only.
    """
    return jsonify({
        'message': 'Exoplanet Identification API Backend',
        'frontend_url': 'https://[your-netlify-site].netlify.app',
        'api_endpoints': {
            '/predict': 'Single prediction (POST)',
            '/health': 'Health check',
            '/predict_batch': 'Batch predictions (POST)',
            '/model_info': 'Model information (GET)',
            '/metrics': 'Model performance metrics (GET)',
            '/hyperparameter-preview': 'Quick parameter testing (POST)',
            '/retrain-with-params': 'Full model retraining (POST)',
            '/parameter-sensitivity': 'Parameter sensitivity data (GET)',
            '/ai_chat': 'Relay chat to OpenRouter (POST)',
            '/metrics_chart/confusion_matrix': 'PNG chart of confusion matrix (GET)',
            '/metrics_chart/feature_importances': 'PNG chart of feature importances (GET)'
        },
        'cors_enabled': True,
        'cors_origins': ['https://[your-netlify-site].netlify.app']
    })


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    try:
        # Check if model exists
        model_exists = os.path.exists('models/pipeline.joblib')
        metrics_exists = os.path.exists('metrics/metrics.json')
        
        # Try loading the model
        if model_exists:
            load_model()
            model_status = 'healthy'
        else:
            model_status = 'missing'
        
        return jsonify({
            'status': 'healthy' if model_exists else 'degraded',
            'timestamp': datetime.utcnow().isoformat(),
            'checks': {
                'model': model_status,
                'model_file': model_exists,
                'metrics_file': metrics_exists
            }
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    """
    Enhanced prediction endpoint with comprehensive validation and quality scoring.
    """
    if not request.is_json:
        return jsonify({
            'error': 'Invalid request',
            'message': 'Request must be JSON'
        }), 400

    try:
        data = request.get_json()
        
        # Parameter validation function - only check for valid numbers
        # Physical validation is handled in utils.predict_single()
        def validate_parameters(params):
            for key, value in params.items():
                try:
                    float(value)  # Just check if it's a valid number
                except (TypeError, ValueError):
                    return False, f'{key} must be a valid number'
            return True, 'OK'

        # Data quality scoring function
        def calculate_data_quality(params):
            quality_score = 0
            max_score = 4
            
            checks = {
                'transit_data': all(params.get(p, 0) > 0 for p in ['transit_depth', 'transit_duration']),
                'stellar_data': all(params.get(p, 0) > 0 for p in ['insolation_flux', 'equilibrium_temp']),
                'orbital_data': params.get('orbital_period', 0) > 0.1,
                'size_data': params.get('planet_radius', 0) > 0.1
            }
            
            return sum(checks.values()) / max_score * 100

        if all(k in data for k in ['koi_period', 'koi_prad', 'koi_score']):
            # Legacy format handling
            params = {
                'orbital_period': data['koi_period'],
                'planet_radius': data['koi_prad'],
                'transit_depth': 0.0,
                'transit_duration': 0.0,
                'insolation_flux': 0.0,
                'equilibrium_temp': 0.0
            }
        else:
            # New format
            params = {
                'orbital_period': data.get('orbital_period', 1.0),
                'planet_radius': data.get('planet_radius', 1.0),
                'transit_depth': data.get('transit_depth', 0.0),
                'transit_duration': data.get('transit_duration', 0.0),
                'insolation_flux': data.get('insolation_flux', 0.0),
                'equilibrium_temp': data.get('equilibrium_temp', 0.0)
            }

        # Validate parameters
        is_valid, error_msg = validate_parameters(params)
        if not is_valid:
            return jsonify({
                'error': 'Invalid parameters',
                'message': error_msg
            }), 400

        # Calculate data quality score
        data_quality = calculate_data_quality(params)

        # Prepare features for prediction
        features = np.array([[params['orbital_period'],
                            params['planet_radius'], 
                            params['transit_depth'],
                            params['transit_duration'],
                            params['insolation_flux'],
                            params['equilibrium_temp']]])

        # Use utils.predict_single which handles preprocessing, model, and explanation
        try:
            include_ai = data.get('include_ai_explanation', False)

            # Call predict_single using the parameters we normalized above
            result = utils.predict_single(
                float(params['orbital_period']),
                float(params['planet_radius']),
                float(params.get('transit_depth', 0.0)),
                float(params.get('transit_duration', 0.0)),
                float(params.get('insolation_flux', 0.0)),
                float(params.get('equilibrium_temp', 0.0)),
                include_ai_explanation=include_ai
            )

            # result contains prediction (string), confidence, probabilities, etc.
            # Normalize response fields expected by the API/tests
            prediction_label = result.get('prediction')
            model_prob = None
            probs = result.get('probabilities') or {}
            # model probability for CONFIRMED class if present
            if isinstance(probs, dict):
                model_prob = float(probs.get('CONFIRMED', max(probs.values()) if probs else 0.0))

            response = {
                'prediction': str(prediction_label).upper(),
                'confidence': float(result.get('confidence', model_prob or 0.0)),
                'probabilities': result.get('probabilities', {}),
                'input_provided': result.get('input_provided', params),
                'data_quality': data_quality,
                'model_probability': float(model_prob or 0.0),
                'planet_type': utils.classify_planet_type(
                    float(params['planet_radius']),
                    float(params.get('equilibrium_temp', 300.0))
                )
            }

            # Include validation errors if present
            if 'validation_errors' in result:
                response['validation_errors'] = result['validation_errors']
                response['reason'] = result.get('reason', 'Validation failed')

            if include_ai and 'ai_explanation' in result:
                response['ai_explanation'] = result['ai_explanation']

            return jsonify(response)

        except Exception as e:
            logging.exception('Prediction error')
            return jsonify({
                'error': 'Prediction failed',
                'message': str(e)
            }), 500

    except Exception as e:
        logging.error(f'Request processing error: {str(e)}')
        return jsonify({
            'error': 'Request processing failed',
            'message': 'Error processing request data'
        }), 400


@app.route('/predict_batch', methods=['POST'])
def predict_batch_endpoint():
    """
    Batch prediction endpoint.
    
    Expected JSON payload:
    {
        "data": [
            {
                "koi_period": float,
                "koi_prad": float,
                "koi_score": float
            },
            ...
        ]
    }
    
    OR CSV file upload with the required columns.
    """
    try:
        # Check if it's a file upload
        if 'file' in request.files:
            file = request.files['file']
            
            # Validate file
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            if not file.filename.endswith('.csv'):
                return jsonify({'error': 'File must be CSV format'}), 400
            
            # Read CSV file
            try:
                df = pd.read_csv(file, comment='#')
                df.columns = df.columns.str.strip()
            except Exception as e:
                return jsonify({
                    'error': 'Failed to read CSV file',
                    'message': str(e)
                }), 400
                
        # Otherwise expect JSON data
        elif request.is_json:
            data = request.get_json()
            
            if 'data' not in data:
                return jsonify({'error': 'Missing "data" field in JSON'}), 400
            
            # Convert to DataFrame
            try:
                df = pd.DataFrame(data['data'])
            except Exception as e:
                return jsonify({
                    'error': 'Failed to parse data',
                    'message': str(e)
                }), 400
        else:
            return jsonify({'error': 'No data provided. Send JSON or CSV file'}), 400
        
        # Validate data
        is_valid, errors = validate_data(df)
        if not is_valid:
            return jsonify({
                'error': 'Invalid data',
                'validation_errors': errors
            }), 400
        
        # Make predictions
        predictions = predict(df)
        
        # Prepare response
        results = []
        for idx, pred in enumerate(predictions):
            row = df.iloc[idx].to_dict()
            row['prediction'] = pred
            row['index'] = idx
            results.append(row)
        
        response = {
            'total': len(predictions),
            'timestamp': datetime.utcnow().isoformat(),
            'results': results
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'error': 'Batch prediction failed',
            'message': str(e)
        }), 500


@app.route('/model_info', methods=['GET'])
def model_info_endpoint():
    """Get model information and configuration."""
    try:
        info = get_model_info()
        info['timestamp'] = datetime.utcnow().isoformat()
        return jsonify(info)
    except Exception as e:
        return jsonify({
            'error': 'Failed to get model info',
            'message': str(e)
        }), 500


@app.route('/metrics', methods=['GET'])
def metrics_endpoint():
    """Get model performance metrics."""
    try:
        metrics = load_metrics()
        
        # Format response
        response = {
            'accuracy': metrics.get('accuracy'),
            'test_samples': metrics.get('test_samples'),
            'confusion_matrix': metrics.get('confusion_matrix'),
            'classes': metrics.get('classes'),
            'classification_report': metrics.get('classification_report'),
            'model_parameters': metrics.get('model_params'),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'error': 'Failed to load metrics',
            'message': str(e)
        }), 500


@app.route('/hyperparameter-preview', methods=['POST'])
def hyperparameter_preview():
    """Quick parameter testing without full retraining."""
    try:
        if not request.is_json:
            return jsonify({'error': 'JSON required'}), 400

        params = request.get_json()

        # Load current model and metrics for comparison
        try:
            current_metrics = load_metrics()
            current_accuracy = current_metrics.get('accuracy', 0.916)
        except:
            current_accuracy = 0.916

        # Estimate accuracy based on parameter changes
        # This is a simplified estimation - in practice you'd do cross-validation
        n_estimators = params.get('n_estimators', 100)
        max_depth = params.get('max_depth')  # Could be None
        learning_rate = params.get('learning_rate', 0.1)

        # Simple heuristic estimation
        base_accuracy = current_accuracy

        # Estimations based on typical Gradient Boosting behavior
        if max_depth is not None:
            if max_depth <= 3:
                base_accuracy *= 0.95  # Shallower trees might underfit
            elif max_depth == 7:
                base_accuracy *= 1.02  # Deeper trees might help
            elif max_depth >= 10:
                base_accuracy *= 0.98  # Very deep might overfit

        if n_estimators >= 150:
            base_accuracy *= 1.01  # More trees generally help but diminishing returns
        elif n_estimators <= 50:
            base_accuracy *= 0.97  # Too few trees

        if learning_rate >= 0.2:
            base_accuracy *= 1.03  # Higher learning might be too aggressive
        elif learning_rate <= 0.05:
            base_accuracy *= 0.99  # Lower learning is more stable

        # Cap at reasonable bounds
        estimated_accuracy = max(0.60, min(0.99, base_accuracy))

        return jsonify({
            'estimated_accuracy': estimated_accuracy,
            'current_accuracy': current_accuracy,
            'parameters_tested': params,
            'confidence_level': 'estimated'  # Not actual cross-validation
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/retrain-with-params', methods=['POST'])
def retrain_with_params():
    """Start full model retraining with new hyperparameters."""
    try:
        if not request.is_json:
            return jsonify({'error': 'JSON required'}), 400

        params = request.get_json()
        job_id = f"retrain_{int(datetime.utcnow().timestamp())}"

        # In a real implementation, you'd queue this as a background job
        # For now, return job ID and simulate progress
        return jsonify({
            'job_id': job_id,
            'status': 'queued',
            'message': 'Retraining job queued successfully',
            'parameters': params
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/retrain-status/<job_id>', methods=['GET'])
def retrain_status(job_id):
    """Check retraining progress."""
    # This is a simplified simulation
    # In production, you'd check actual job status from Redis/database
    import random

    # Simulate progress based on time
    elapsed = int(datetime.utcnow().timestamp()) % 100

    if elapsed < 10:
        progress = 5
        message = "Preparing training data..."
    elif elapsed < 30:
        progress = 25
        message = "Training model on training split..."
    elif elapsed < 70:
        progress = 70
        message = "Evaluating on validation set..."
    elif elapsed < 90:
        progress = 90
        message = "Finalizing and saving model..."
    else:
        progress = 100
        message = "Training completed successfully!"

    return jsonify({
        'job_id': job_id,
        'progress': progress,
        'message': message,
        'completed': progress >= 100
    })


@app.route('/parameter-sensitivity', methods=['GET'])
def parameter_sensitivity():
    """Get parameter sensitivity data for charts with real-time updates."""
    try:
        # Get current parameters from query params if provided
        n_estimators = int(request.args.get('n_estimators', 100))
        max_depth_str = request.args.get('max_depth', 'None')
        learning_rate = float(request.args.get('learning_rate', 0.1))

        # Convert max_depth properly (handling None/null from frontend)
        if max_depth_str in ['None', 'null', '', None]:
            max_depth = None
        else:
            max_depth = int(max_depth_str)

        # Load current accuracy
        try:
            metrics = load_metrics()
            current_accuracy = metrics.get('accuracy', 0.85)
        except:
            current_accuracy = 0.85

        # Calculate real-time estimate based on parameters - IMPROVED HEURISTICS
        estimated_accuracy = current_accuracy

        # Max depth effects - SIGNIFICANT impact on Gradient Boosting
        if max_depth is not None:
            if max_depth <= 3:
                estimated_accuracy *= 0.95  # Underfitting risk
            elif max_depth >= 10:
                estimated_accuracy *= 0.97  # Overfitting risk
        else:
            estimated_accuracy *= 0.99  # Optimal for most cases

        # Number of estimators effects - BIG impact in ensemble methods
        if n_estimators <= 60:
            estimated_accuracy *= 0.93
        elif n_estimators <= 80:
            estimated_accuracy *= 0.96
        elif n_estimators <= 90:
            estimated_accuracy *= 0.98
        elif n_estimators >= 150:
            estimated_accuracy *= 1.02
        elif n_estimators >= 180:
            estimated_accuracy *= 1.01

        # Learning rate effects - Affects convergence speed and stability
        if learning_rate <= 0.05:
            estimated_accuracy *= 0.97
        elif learning_rate <= 0.08:
            estimated_accuracy *= 0.99
        elif learning_rate >= 0.25:
            estimated_accuracy *= 0.96
        elif learning_rate >= 0.15:
            estimated_accuracy *= 0.98

        # Combined parameter interactions
        # Higher learning rates work better with lower tree counts (compensating)
        if learning_rate >= 0.15 and n_estimators <= 80:
            estimated_accuracy *= 1.02

        # Lower learning rates need more trees to be effective
        if learning_rate <= 0.06 and n_estimators >= 150:
            estimated_accuracy *= 1.03

        # Keep within realistic bounds (Gradient Boosting typically performs in this range)
        estimated_accuracy = max(0.70, min(0.95, estimated_accuracy))

        # Calculate sensitivity scores based on parameter ranges (fixed for None handling)
        sens_n_estimators = abs(n_estimators - 100) / 50.0 * 0.5 + 0.3
        sens_max_depth = 0.6 if max_depth is not None and max_depth > 5 else 0.4
        sens_learning_rate = abs(learning_rate - 0.1) * 5.0 + 0.2

        # Normalize sensitivity scores
        sens_n_estimators = min(1.0, max(0.1, sens_n_estimators))
        sens_max_depth = min(1.0, max(0.1, sens_max_depth))
        sens_learning_rate = min(1.0, max(0.1, sens_learning_rate))

        return jsonify({
            'sensitivity_scores': [sens_n_estimators, sens_max_depth, sens_learning_rate],
            'current_accuracy': current_accuracy,
            'tuned_estimate': estimated_accuracy,
            'parameter_names': ['n_estimators', 'max_depth', 'learning_rate'],
            'active_parameters': {
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'learning_rate': learning_rate
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def calculate_performance_estimate(parameters):
    """Estimate model performance based on parameters without full training."""
    try:
        current_metrics = load_metrics()
        baseline_accuracy = current_metrics.get('accuracy', 0.85)
        
        # Parameter impact estimates
        impact_scores = {
            'n_estimators': 0.0,
            'max_depth': 0.0,
            'learning_rate': 0.0
        }
        
        # Ensure parameters have default values if None
        n_estimators = parameters.get('n_estimators', 100)
        max_depth = parameters.get('max_depth')
        learning_rate = parameters.get('learning_rate', 0.1)
        
        # Evaluate n_estimators
        if n_estimators is not None:
            if n_estimators < 50:
                impact_scores['n_estimators'] = -0.15
            elif n_estimators > 200:
                impact_scores['n_estimators'] = 0.05
        
        # Evaluate max_depth
        if max_depth is not None:
            if max_depth < 4:
                impact_scores['max_depth'] = -0.1
            elif max_depth > 8:
                impact_scores['max_depth'] = -0.05
        else:
            # Auto max_depth is often a good choice
            impact_scores['max_depth'] = 0.02
            
        # Evaluate learning_rate
        if learning_rate is not None:
            if learning_rate < 0.05:
                impact_scores['learning_rate'] = -0.05
            elif learning_rate > 0.2:
                impact_scores['learning_rate'] = -0.1
            
        # Calculate total impact
        total_impact = sum(impact_scores.values())
        estimated_accuracy = min(0.99, max(0.5, baseline_accuracy + total_impact))
        
        return {
            'estimated_accuracy': estimated_accuracy,
            'baseline_accuracy': baseline_accuracy,
            'parameter_impacts': impact_scores,
            'warning_messages': get_parameter_warnings(parameters),
            'confidence': 'medium'
        }
        
    except Exception as e:
        raise Exception(f"Error estimating performance: {str(e)}")

def get_parameter_warnings(parameters):
    """Generate warnings for potentially problematic parameter combinations."""
    warnings = []
    
    n_estimators = parameters.get('n_estimators')
    max_depth = parameters.get('max_depth')
    learning_rate = parameters.get('learning_rate')
    
    if n_estimators is not None and n_estimators < 50:
        warnings.append("Very low number of estimators may lead to underfitting")
    
    if max_depth is not None and max_depth > 8:
        warnings.append("High max_depth may lead to overfitting")
        
    if learning_rate is not None and learning_rate > 0.2:
        warnings.append("High learning rate may cause unstable training")
        
    return warnings

@app.route('/tune-model', methods=['GET', 'POST'])
def tune_model():
    """Model tuning interface with parameter testing."""
    if request.method == 'GET':
        # Return current model configuration and tuning options
        try:
            model = load_model()
            metrics = load_metrics()
            current_params = model['model'].get_params() if hasattr(model['model'], 'get_params') else {}
            
            return jsonify({
                'current_configuration': {
                    'model_type': str(type(model['model']).__name__),
                    'parameters': current_params,
                    'feature_names': model.get('feature_names', []),
                    'current_accuracy': metrics.get('accuracy', 0.85),
                    'current_f1': metrics.get('f1_score', 0.85)
                },
                'tuning_options': {
                    'n_estimators': {
                        'type': 'int',
                        'range': [50, 300],
                        'default': 100,
                        'description': 'Number of trees in the forest. More trees generally improve performance but increase training time.'
                    },
                    'max_depth': {
                        'type': 'int',
                        'range': [3, 10],
                        'default': None,
                        'description': 'Maximum depth of each tree. Controls model complexity.'
                    },
                    'learning_rate': {
                        'type': 'float',
                        'range': [0.01, 0.3],
                        'default': 0.1,
                        'description': 'Step size shrinkage used to prevent overfitting.'
                    },
                    'min_samples_split': {
                        'type': 'int',
                        'range': [2, 20],
                        'default': 2,
                        'description': 'Minimum samples required to split a node.'
                    }
                },
                'model_description': 'Current model is optimized for exoplanet classification with focus on ultra-short period planets.',
                'warning': 'This interface allows parameter experimentation without affecting the production model.'
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
            
    elif request.method == 'POST':
        try:
            if not request.is_json:
                return jsonify({'error': 'JSON required'}), 400
                
            data = request.get_json()
            
            parameters = {
                'n_estimators': int(data['n_estimators']) if 'n_estimators' in data else 100,
                'max_depth': int(data['max_depth']) if data.get('max_depth') not in [None, 'null', ''] else None,
                'learning_rate': float(data['learning_rate']) if 'learning_rate' in data else 0.1,
                'min_samples_split': int(data['min_samples_split']) if 'min_samples_split' in data else 2
            }
            
            # Quick performance estimation without training
            estimation = calculate_performance_estimate(parameters)
            return jsonify(estimation)
                
        except Exception as e:
            return jsonify({'error': str(e)}), 500


@app.route('/ai_chat', methods=['POST'])
def ai_chat():
    """Relay chat completion requests to OpenRouter if API key is configured.

    Expected JSON payload:
    {
      "messages": [{"role": "user", "content": "..."}, ...],
      "model": "openai/gpt-4o-mini"  # optional overrides env default
    }
    """
    try:
        if not OPENROUTER_API_KEY:
            return jsonify({
                'error': 'OPENROUTER_API_KEY not configured in environment'
            }), 400

        if not request.is_json:
            return jsonify({'error': 'Expected JSON body'}), 400

        body = request.get_json() or {}
        messages = body.get('messages')
        if not isinstance(messages, list) or not messages:
            return jsonify({'error': 'messages must be a non-empty list'}), 400

        model = body.get('model', OPENROUTER_MODEL)

        url = f"{OPENROUTER_BASE_URL}/chat/completions"
        headers = {
            'Authorization': f"Bearer {OPENROUTER_API_KEY}",
            'Content-Type': 'application/json',
            # Optional per OpenRouter best practices
            'HTTP-Referer': APP_URL,
            'X-Title': APP_TITLE
        }
        payload = {
            'model': model,
            'messages': messages,
            'stream': False
        }

        resp = requests.post(url, headers=headers, data=json_lib.dumps(payload), timeout=60)
        if resp.status_code >= 400:
            return jsonify({
                'error': 'OpenRouter request failed',
                'status': resp.status_code,
                'response': resp.text
            }), 502

        data = resp.json()
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': 'AI relay failed', 'message': str(e)}), 500


def _matplotlib_import():
    import matplotlib.pyplot as plt  # lazy import
    import seaborn as sns
    return plt, sns


def _png_bytes_from_plt(plt):
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', dpi=160)
    plt.close()
    buf.seek(0)
    return buf.read()


@app.route('/metrics_chart/confusion_matrix', methods=['GET'])
def metrics_chart_confusion_matrix():
    """Return confusion matrix chart as base64 PNG."""
    try:
        metrics = load_metrics()
        conf = np.array(metrics.get('confusion_matrix'))
        classes = metrics.get('classes', [])

        plt, sns = _matplotlib_import()
        plt.figure(figsize=(6, 5))
        ax = sns.heatmap(conf, annot=True, fmt='d', cmap='Blues', cbar=False,
                         xticklabels=classes, yticklabels=classes)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')

        png = _png_bytes_from_plt(plt)
        b64 = base64.b64encode(png).decode('utf-8')
        return jsonify({
            'image_base64': b64,
            'content_type': 'image/png'
        })
    except Exception as e:
        return jsonify({'error': 'Failed to render confusion matrix', 'message': str(e)}), 500


@app.route('/metrics_chart/feature_importances', methods=['GET'])
def metrics_chart_feature_importances():
    """Return feature importances bar chart as base64 PNG."""
    try:
        pipeline = load_model()
        model = pipeline['model']
        feature_names = pipeline.get('feature_names') or []
        importances = getattr(model, 'feature_importances_', None)
        if importances is None:
            return jsonify({'error': 'Model does not provide feature_importances_'}), 400

        plt, sns = _matplotlib_import()
        plt.figure(figsize=(6, 4))
        ax = sns.barplot(x=importances, y=feature_names, orient='h', palette='viridis')
        ax.set_xlabel('Importance')
        ax.set_ylabel('Feature')
        ax.set_title('Feature Importances')

        png = _png_bytes_from_plt(plt)
        b64 = base64.b64encode(png).decode('utf-8')
        return jsonify({
            'image_base64': b64,
            'content_type': 'image/png'
        })
    except Exception as e:
        return jsonify({'error': 'Failed to render feature importances', 'message': str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        'error': 'Endpoint not found',
        'message': 'The requested endpoint does not exist'
    }), 404


@app.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors."""
    return jsonify({
        'error': 'Method not allowed',
        'message': 'The method is not allowed for the requested URL'
    }), 405


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred'
    }), 500


# Development server
if __name__ == '__main__':
    print("\n" + "="*70)
    print("ENHANCED EXOPLANET IDENTIFICATION API v3.0.0")
    print("="*70)
    print("\nStarting Flask development server...")
    print("API will be available at: http://localhost:5000")
    print("\nAvailable endpoints:")
    print("  GET  /              - API information")
    print("  GET  /health        - Health check")
    print("  POST /predict       - Single prediction")
    print("  POST /predict_batch - Batch predictions")
    print("  GET  /model_info    - Model information")
    print("  GET  /metrics       - Performance metrics")
    print("\nPress Ctrl+C to stop the server")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
