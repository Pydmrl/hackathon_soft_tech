"""
Flask Web Application - Real Estate Investment Advisor
Model v7 ile profesyonel mimari
"""
import os
import sys
import joblib
from flask import Flask
from flask_cors import CORS

# Konfigürasyon
from config import config
from services import ModelService, PredictionService
from api import api_bp, init_routes


def create_app(config_name='development'):
    """
    Flask uygulaması factory pattern ile oluştur
    
    Args:
        config_name: 'development' veya 'production'
        
    Returns:
        Flask app instance
    """
    # Flask app
    app = Flask(__name__)
    
    # Konfigürasyon yükle
    app.config.from_object(config[config_name])
    
    # CORS aktif et
    CORS(app, resources={r"/*": {"origins": app.config['CORS_ORIGINS']}})
    
    # Model klasörünü oluştur
    models_dir = os.path.join(app.config['BASE_DIR'], 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Logs klasörünü oluştur
    logs_dir = os.path.join(app.config['BASE_DIR'], 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    # Servisleri initialize et
    print("\n" + "="*80)
    print("🏠 REAL ESTATE INVESTMENT ADVISOR v7 - AI_Spark_Team")
    print("="*80 + "\n")
    
    try:
        # Model servisini başlat
        model_service = ModelService(
            model_path=app.config['MODEL_PATH'],
            location_path=app.config['LOCATION_DATA_PATH']
        )
        
        # Model data'yı yükle (prediction service için)
        model_data = joblib.load(app.config['MODEL_PATH'])
        
        # Prediction servisini başlat
        prediction_service = PredictionService(model_data)
        
        # Metrikleri yazdır
        metrics = prediction_service.get_metrics()
        print("\n📊 Model Performance:")
        print(f"   Test MAPE: {metrics.get('mape', 0):.2f}%")
        print(f"   Test R²: {metrics.get('r2', 0):.4f}")
        
        # API routes'ları initialize et (servisleri global değişkenlere ata)
        # ÖNCE servisleri global yap, SONRA blueprint register et
        init_routes(model_service, prediction_service)
        
        # Blueprint'i register et (artık model_service hazır)
        app.register_blueprint(api_bp)
        
        print("\n✅ Application initialized successfully!")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        print("Application failed to initialize!")
        sys.exit(1)
    
    return app


def run_server(host='0.0.0.0', port=5000, debug=True):
    """
    Development server'ı çalıştır
    
    Args:
        host: Server host
        port: Server port
        debug: Debug mode
    """
    app = create_app('development' if debug else 'production')
    
    print(f"🌐 Starting server on http://{host}:{port}")
    print(f"🔍 Debug mode: {'ON' if debug else 'OFF'}")
    print("="*80 + "\n")
    
    # MemoryError'u önlemek için use_reloader=False
    app.run(host=host, port=port, debug=debug, use_reloader=False)


if __name__ == '__main__':
    # Ortam değişkenlerinden veya varsayılan değerlerden çalıştır
    env = os.environ.get('FLASK_ENV', 'development')
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', 5000))
    debug = env == 'development'
    
    run_server(host=host, port=port, debug=debug)
