# Trinetra AI — Misinformation Detection

Trinetra AI is a creative solution brought to the market that produces verifications for AI-created text or other conceivable falsehoods. It supplies a quick and accessible web experience where users can receive the detected verdicts, confidence scores, and topical categories along with graphics of trend and community feedback. It also provides a popular front end **React** for user interactions engaging with an API sandbox for **Gemini-driven analysis**. It includes **MongoDB** as a database to help further users monitor and follow trends and receive feedback from the community.

---

## What it does

- In real-time, it verifies the users' text and provides a sound judgment from the AI system: **Misinformation** or **Reliable**.  
- The tool features a confidence aspect showing how confident the AI is in the particular decision and assigns a category to indicate the nature of the text.  
- App keeps the record of objects flagged by users for later auditing, analytics, and continuous improvement through deep learning models.  
- Community members can vote on particular cases to enhance the visibility of those most interesting under future moderation and help with potential impact.  
- Those interested in the system's health can use the data model provided to prepare the dashboard for category summaries and insights.  

---

## Why it matters

- Before users can disseminate the content, it quickly and easily helps in detangling the web of misinformation to make it more trustworthy.  
- Admins and researchers move several steps ahead in spotting misinformation trends, and now they can track the psyches through themes like **health, politics, and climate**.  
- This foundation also makes it possible to create **hybrid ML models** that combine LLM reasoning.  

---

## Core features

- **Gemini** powers the AI-only decision path for uniform, quick answers with minimal latency.  
- New users get immediate feedback on the accuracy of the presented statement along with the conclusion, confidence level, category, and explanation, all of this wrapped up in a simple UI.  
- The storage of **MongoDB** is supported with historical analysis, and the implementation of item searches in dashboards is forged by the metadata.  
- The **upvote** allows for the community to contribute their discernment by ranking the priority of review.  
- The application incorporates a number of resilient features, like **fallbacks and error handling**, which are meant to allow users to have the best experience even in the presence of API difficulties.  

---

## Technology

- **Frontend:** React, Vite, Tailwind CSS — fast and responsive UI with modular components.  
- **Backend:** Node.js with Express — API orchestration, validation, and storage.  
- **AI:** Gemini API — designed for identification of misinformation, outputs structured JSON formats.  
- **Database:** MongoDB — stores results, categories, confidence, timestamps, and votes.  
- **Optional ML service:** Future hybrid inference utilizing Python (FastAPI) retained for scaffolding.  

---

## Roadmap

- **Hybrid inference:** Mix LLM reasoning with custom classifiers for more accurate results.  
- **Source/context enrichment:** Citations, claim matching, and relevant fact-checking.  
- **Role-based dashboards:** Moderator tools, analytics views, and exportable reports.  
- **Trust signals:** Model lineage, prompt transparency, and evaluation benchmarks.  

---

## Ethics and usage

Trinetra AI is an assistive tool, not the final decider of truth. Always verify with multiple trusted sources and include human oversight when it comes to crucial info.

---
## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

- Inspired by popular mininformation detectors and classifiers.
- Thanks to all contributors and the open-source community.

---
**Maintainer:** [jeevapriyan10](https://github.com/jeevapriyan10)
