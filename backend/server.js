const express = require('express');
const cors = require('cors');
const { MongoClient } = require('mongodb');
const axios = require('axios');
const fs = require('fs');
const path = require('path');
//const fetch = require('node-fetch');
const yauzl = require('yauzl');
const mkdirp = require('mkdirp');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 8000;

app.use(cors());
app.use(express.json());

let db;

/* --------------------- MongoDB --------------------- */
const connectDB = async () => {
  try {
    const client = new MongoClient(process.env.MONGO_URI);
    await client.connect();
    db = client.db(process.env.MONGO_DB_NAME);
    console.log('✅ Connected to MongoDB');
  } catch (error) {
    console.error('❌ MongoDB connection error:', error.message);
  }
};

/* --------------------- Model Setup --------------------- */
const setupModel = async () => {
  const modelDir = path.join(__dirname, 'models');
  const modelZipPath = path.join(__dirname, 'model.zip');

  if (!fs.existsSync(modelDir)) {
    console.log('📦 Model directory not found. Downloading model...');

    try {
      const modelUrl = process.env.MODEL_ZIP_URL;
      if (!modelUrl) {
        console.log('⚠️ MODEL_ZIP_URL not configured, skipping model download');
        return;
      }

      const response = await fetch(modelUrl);
      const buffer = await response.buffer();
      fs.writeFileSync(modelZipPath, buffer);

      await mkdirp(modelDir);
      await new Promise((resolve, reject) => {
        yauzl.open(modelZipPath, { lazyEntries: true }, (err, zipfile) => {
          if (err) reject(err);

          zipfile.readEntry();
          zipfile.on('entry', (entry) => {
            if (/\/$/.test(entry.fileName)) {
              zipfile.readEntry();
            } else {
              zipfile.openReadStream(entry, (err, readStream) => {
                if (err) reject(err);

                const filePath = path.join(modelDir, entry.fileName);
                mkdirp(path.dirname(filePath)).then(() => {
                  readStream.pipe(fs.createWriteStream(filePath));
                  readStream.on('end', () => zipfile.readEntry());
                });
              });
            }
          });

          zipfile.on('end', () => {
            fs.unlinkSync(modelZipPath);
            console.log('✅ Model extracted successfully');
            resolve();
          });
        });
      });
    } catch (error) {
      console.error('❌ Error setting up model:', error.message);
    }
  }
};

/* --------------------- ML Microservice --------------------- */
const callMLService = async (text) => {
  try {
    await axios.post('http://localhost:8001/predict', { text }, { timeout: 5000 });
    return null;
  } catch (error) {
    console.log('⚠️ ML microservice not available:', error.message);
    return null;
  }
};

/* --------------------- Gemini API --------------------- */
const callGeminiAPI = async (text) => {
  try {
    const response = await axios.post(
      `https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=${process.env.GEMINI_API_KEY}`,
      {
        contents: [
          {
            parts: [
              {
                text: `Analyze this text for misinformation. Respond strictly with JSON: {"is_misinformation": boolean, "confidence": number (0-1), "category": "string", "explanation": "string"}. Text: "${text}"`,
              },
            ],
          },
        ],
      },
      { headers: { "Content-Type": "application/json" } }
    );

    const content = response.data.candidates[0].content.parts[0].text;

    const jsonMatch = content.match(/\{[\s\S]*\}/);
    if (jsonMatch) {
      return JSON.parse(jsonMatch[0]);
    }

    throw new Error("Invalid response format from Gemini");
  } catch (error) {
    console.error("❌ Gemini API error:", error.response?.data || error.message);
    return {
      is_misinformation: false,
      confidence: 0.0,
      category: "unknown",
      explanation: "Unable to analyze due to API error",
    };
  }
};


/* --------------------- Routes --------------------- */
app.post('/api/verify', async (req, res) => {
  try {
    const { text } = req.body;

    if (!text) {
      return res.status(400).json({ error: 'Text is required' });
    }

    const geminiResult = await callGeminiAPI(text);

    let finalDecision = geminiResult.is_misinformation || false;
    let confidence = geminiResult.confidence || 0.5;
    let category = geminiResult.category || 'unknown';

    if (db && finalDecision) {
      await db.collection('misinformation').insertOne({
        text,
        category,
        confidence,
        timestamp: new Date(),
        upvotes: 0
      });
    }

    res.json({
      ml_result: null,
      gemini_result: geminiResult,
      final_decision: finalDecision,
      confidence,
      category
    });
  } catch (error) {
    console.error('❌ Verification error:', error.message);
    res.status(500).json({ error: 'Internal server error' });
  }
});

app.get('/api/dashboard', async (req, res) => {
  try {
    if (!db) return res.status(500).json({ error: 'Database not connected' });

    const categorySummary = await db.collection('misinformation').aggregate([
      {
        $group: {
          _id: '$category',
          count: { $sum: 1 },
          texts: {
            $push: {
              text: '$text',
              upvotes: '$upvotes',
              confidence: '$confidence',
              timestamp: '$timestamp'
            }
          }
        }
      }
    ]).toArray();

    res.json(categorySummary);
  } catch (error) {
    console.error('❌ Dashboard error:', error.message);
    res.status(500).json({ error: 'Internal server error' });
  }
});

app.post('/api/upvote', async (req, res) => {
  try {
    const { category, text } = req.body;
    if (!category || !text) {
      return res.status(400).json({ error: 'Category and text are required' });
    }

    if (!db) return res.status(500).json({ error: 'Database not connected' });

    await db.collection('misinformation').updateOne(
      { category, text },
      { $inc: { upvotes: 1 } }
    );

    res.json({ success: true });
  } catch (error) {
    console.error('❌ Upvote error:', error.message);
    res.status(500).json({ error: 'Internal server error' });
  }
});

app.get('/health', (req, res) => {
  res.json({ status: 'OK', timestamp: new Date().toISOString() });
});

/* --------------------- Start Server --------------------- */
const startServer = async () => {
  await connectDB();
  await setupModel();

  app.listen(PORT, () => {
    console.log(`🚀 Backend server running on port ${PORT}`);
  });
};

startServer().catch(console.error);
