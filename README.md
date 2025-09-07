# Gemini K8s Agent

A simple Python web app that integrates Google Gemini with MCP (Model Context Protocol) to answer and perform Kubernetes queries in natural language. It acts as an AI agent, connecting to your local Kubernetes cluster through MCP without writing direct API calls.

---

## Features
- 🌐 Flask-based frontend for natural language queries
- 🤖 Gemini AI backend agent
- 🔗 MCP integration to access Kubernetes cluster
- ⚡ No manual API coding required

---

## Architecture
User → Flask Web App → Gemini Agent → MCP Server → Kubernetes Cluster

---

## Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/gemini-k8s-agent.git
cd gemini-k8s-agent

### 2. Setup environment

Install dependencies using uv
uv venv
uv sync

Create .env:
GEMINI_API_KEY=your_api_key_here

3. Run MCP Server

Run or configure your Kubernetes MCP server to connect to your cluster.

4. Start the Flask app
python app.py

Visit: http://127.0.0.1:5000