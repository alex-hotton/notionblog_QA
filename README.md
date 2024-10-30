# NotionDB Q&A System

## Overview
A powerful search interface for Notion databases that can be embedded directly into Replit blocks and Notion blogs. This project enables natural language querying of your Notion database content using advanced RAG (Retrieval-Augmented Generation) technology.

## Current Features
- RAG (Retrieval-Augmented Generation) pipeline integrated with Notion databases
- GPT-4 Mini models support for intelligent query processing
- Streamlit-based user interface

## Roadmap
1. Multi-LLM Support
   - Integration of various language models
   - Support for multimodal capabilities
2. Source Attribution
   - Clear referencing of information sources from the database
3. UI/UX Enhancements
   - Improved interface design
   - Better user experience
4. Feedback System
   - User feedback collection
   - Rating system for responses
5. Analytics & Storage
   - SQLite database implementation
   - Storage of predictions and feedback
   - Performance analytics
6. Smart Embedding Updates
   - Intelligent updating mechanism
   - Efficient resource utilization

## Quick Start Guide
1. Clone the repository:
   ```bash
   git clone https://github.com/alex-hotton/notiondb_QA.git
   ```

2. Set up environment variables:
   - Rename `.env.example` to `.env`
   - Add your API keys to the `.env` file

3. Run the application:
   ```bash
   streamlit run main.py
   ```

4. Optional: Deploy on Replit
   - Deploy the application on Replit
   - Embed the Replit block in your Notion blog

## Contributing
We welcome contributions! If you have feature requests or suggestions, please feel free to:
- Open an issue
- Submit a pull request
- Reach out with your ideas
