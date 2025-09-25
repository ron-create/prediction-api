# Dengue Risk Prediction API

A FastAPI backend system for predicting dengue risk in Dasmariñas, Cavite using machine learning, weather data, and geographic information.

## 🚀 Features

- **Machine Learning Prediction**: Logistic Regression model trained on synthetic data
- **Weather Integration**: Real-time weather data from OpenWeatherMap API
- **Geographic Analysis**: GeoJSON-based barangay and waterway analysis
- **Database Integration**: Supabase for dengue case data
- **RESTful API**: Complete FastAPI endpoints with Swagger documentation

## 📁 Project Structure

```
dengue-ml/
├── app/
│   ├── main.py              # FastAPI entry point
│   ├── model.py             # ML training & prediction
│   ├── utils.py             # GeoJSON helpers & distance calculations
│   ├── supabase_client.py   # Supabase database integration
│   └── weather_client.py    # Weather API integration
├── data/
│   ├── dasmabarangays.geojson
│   └── waterway.geojson
├── requirements.txt
└── README.md
```

## 🛠️ Installation

1. **Clone and navigate to the project:**
   ```bash
   cd dengue-ml
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   ```

3. **Activate virtual environment:**
   ```bash
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Set up environment variables (optional):**
   Create a `.env` file with:
   ```
   SUPABASE_URL=https://your-project.supabase.co
   SUPABASE_KEY=your-anon-key
   OPENWEATHER_API_KEY=your-openweather-api-key
   ```

## 🚀 Running the API

```bash
uvicorn app.main:app --reload
```

The API will be available at:
- **API**: http://127.0.0.1:8000
- **Documentation**: http://127.0.0.1:8000/docs
- **ReDoc**: http://127.0.0.1:8000/redoc

## 📊 API Endpoints

### Core Endpoints

- `GET /` - Health check
- `GET /health` - Detailed health check
- `POST /predict` - Main prediction endpoint

### Utility Endpoints

- `GET /barangays` - List all barangays
- `GET /model/info` - ML model information
- `GET /weather/{lat}/{lng}` - Weather data for coordinates
- `GET /cases/{barangay}` - Case data for barangay

## 🔬 Example Usage

### Prediction Request
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "lat": 14.329,
       "lng": 120.936
     }'
```

### Response
```json
{
  "risk": "high",
  "probability": 0.78,
  "barangay": "San Simon",
  "features": {
    "rainfall": 25.0,
    "humidity": 88.0,
    "temperature": 29.0,
    "case_density": 0.4,
    "waterway_distance": 120.0
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

## 🧠 Machine Learning Model

- **Algorithm**: Logistic Regression
- **Features**: rainfall, humidity, temperature, case_density, waterway_distance
- **Training**: Synthetic data generation with realistic distributions
- **Evaluation**: Train/test split with accuracy metrics

## 🌐 External Integrations

### Supabase
- Fetches dengue case reports
- Calculates case density per barangay
- Falls back to dummy data if not configured

### OpenWeatherMap
- Real-time weather data
- Rainfall, humidity, temperature
- Falls back to dummy data if not configured

## 🗺️ Geographic Data

- **Barangays**: Dasmariñas barangay boundaries
- **Waterways**: River and creek data
- **Distance Calculation**: Nearest waterway distance
- **Boundary Checking**: Point-in-polygon validation

## 🧪 Testing

Use the Swagger UI at `/docs` to test all endpoints interactively.

## 🔧 Configuration

The system works with dummy data by default. To use real data:

1. Set up Supabase project and configure credentials
2. Get OpenWeatherMap API key
3. Update environment variables

## 📝 Notes

- The ML model uses synthetic training data
- All external APIs have fallback dummy data
- GeoJSON files are loaded on startup
- CORS is enabled for frontend integration

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.
