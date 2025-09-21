# Misinformation Detection App

A full-stack application for detecting misinformation using machine learning models and AI analysis.

## Architecture

- **Backend**: Node.js with Express (Port 8000)
- **ML Service**: Python FastAPI microservice (Port 8001)
- **Frontend**: React with Vite and Tailwind CSS
- **Database**: MongoDB

## Features

- Real-time text verification using ML models and Gemini AI
- Dashboard with misinformation categories and statistics
- Community upvoting system
- Responsive and accessible UI
- Automatic model downloading and setup

## Prerequisites

- Node.js (v16 or higher)
- Python (v3.8 or higher)
- MongoDB (local or cloud instance)
- Gemini API key

## Quick Start

1. **Clone and install dependencies**:
   ```bash
   git clone <repository-url>
   cd misinformation-detection-app
   npm run install:all
   ```

2. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Start all services**:
   ```bash
   npm run dev
   ```

This will start:
- Backend API server on http://localhost:8000
- ML microservice on http://localhost:8001
- Frontend development server on http://localhost:5173

## Environment Variables

Create a `.env` file in the root directory:

```env
# MongoDB Configuration
MONGO_URI=mongodb://localhost:27017
MONGO_DB_NAME=misinformation_db

# Gemini API Configuration
GEMINI_API_KEY=your_gemini_api_key_here

# ML Model Configuration (optional)
MODEL_ZIP_URL=https://example.com/path/to/your/model.zip

# Server Configuration
PORT=8000
```

## API Endpoints

### Backend (Port 8000)

- `POST /api/verify` - Verify text for misinformation
- `GET /api/dashboard` - Get misinformation statistics
- `POST /api/upvote` - Upvote misinformation detection
- `GET /health` - Health check

### ML Service (Port 8001)

- `POST /predict` - Get ML model prediction
- `GET /health` - Health check
- `GET /` - Service info

## Project Structure

```
misinformation-detection-app/
├── backend/                 # Node.js Express API
│   ├── server.js           # Main server file
│   └── package.json        # Backend dependencies
├── ml_service/             # Python FastAPI microservice
│   ├── main.py            # ML service main file
│   └── requirements.txt   # Python dependencies
├── frontend/              # React frontend
│   ├── src/
│   │   ├── components/    # React components
│   │   ├── App.jsx       # Main app component
│   │   └── main.jsx      # Entry point
│   ├── index.html        # HTML template
│   └── package.json      # Frontend dependencies
├── .env.example          # Environment variables template
├── .gitignore           # Git ignore rules
└── README.md           # This file
```

## Development

### Running Individual Services

**Backend only**:
```bash
npm run backend:dev
```

**Frontend only**:
```bash
npm run frontend:dev
```

**ML Service only**:
```bash
npm run ml:dev
```

### Building for Production

```bash
npm run build
```

## Deployment

### Using Docker (Recommended)

1. Create `Dockerfile` for each service
2. Use `docker-compose.yml` for orchestration
3. Deploy to your preferred cloud platform

### Manual Deployment

1. **Backend**: Deploy to Render, Railway, or similar Node.js hosting
2. **ML Service**: Deploy to platforms supporting Python (Render, Railway, etc.)
3. **Frontend**: Build and deploy to Netlify, Vercel, or similar static hosting
4. **Database**: Use MongoDB Atlas or similar cloud database

### Environment Setup for Production

- Set all environment variables in your deployment platform
- Ensure MongoDB is accessible from your backend
- Configure CORS settings for production domains
- Set up proper logging and monitoring

## Model Integration

The application supports custom ML models:

1. Train your misinformation detection model
2. Export as a zip file containing model files
3. Host the zip file on a public URL
4. Set `MODEL_ZIP_URL` in your environment variables
5. The application will automatically download and extract the model

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Support

For issues and questions:
1. Check the GitHub issues
2. Create a new issue with detailed description
3. Include logs and environment details

## Security

- Never commit API keys or sensitive data
- Use environment variables for all secrets
- Regularly update dependencies
- Follow security best practices for production deployment