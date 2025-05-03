# 🚀 E-commerce Analytics & Recommendation System

This project is a **Flask-based web application** containerized using **Docker** that performs customer analytics, product recommendations, and insightful data visualizations using advanced machine learning and data analysis techniques.

---

## 🐳 Docker-Powered Deployment

This project marks my first experience with Docker, and it's fully containerized for seamless deployment and scaling.

### 🛠 Technologies & Libraries Used

#### Core Backend
- **Flask** – Web framework for backend logic and API endpoints
- **SQLite3** – Lightweight database for storage
- **Pandas** – For data manipulation and analysis
- **Threading** – Ensures thread-safe concurrent database access

#### Data Visualization
- **Matplotlib** – Plotting library for all visuals
- **Squarify** – Treemap visualizations
- **Pyplot** – Submodule of Matplotlib for plots
- **Base64** – For embedding plots as images in HTML

#### Machine Learning & Analysis
- **TextBlob** – Sentiment analysis on product descriptions and reviews
- **Custom ML Recommender** – Content-based product recommendation system
- **Contextlib** – Efficient resource handling
- **GC** – Memory management during processing

#### Data Processing
- **Pandas**, **SQLite3**, **IO**, **BytesIO** – For reading, writing, and processing data streams

#### Containerization & Deployment
- **Docker** – Containerization of the application
- **Docker Compose** – For service orchestration

---

## 🤖 Machine Learning Features

### Sentiment Analysis
- Based on **TextBlob**
- Analyzes product descriptions and customer reviews
- Outputs: sentiment polarity and subjectivity
- Used for insights and personalized recommendations

### Recommendation System
- **Content-based filtering** approach
- Utilizes product categories, user preferences, and historical behavior
- Outputs tailored product suggestions for each user

---

## 📊 Advanced Analytics Modules

### Customer Segmentation
- Categorization: Low, Medium, High value customers
- Behavior pattern recognition
- Purchase frequency and value analysis

### Sales Analytics
- Monthly sales trends via time-series plots
- Pie chart for payment mode distribution
- Regional performance breakdown
- Profit tracking by category and subcategory

### Product Analytics
- Top product identification
- Category and subcategory analysis with treemaps
- Tracks Average Order Value (AOV)

---

## 🧪 RESTful API Endpoints

| Endpoint             | Method | Description                             |
|---------------------|--------|-----------------------------------------|
| `/recommend`        | GET    | Get product recommendations             |
| `/sentiment`        | POST   | Analyze sentiment of product review     |
| `/analytics/sales`  | GET    | View sales data visualizations          |
| `/analytics/customer`| GET   | Get customer segmentation data          |


### 🏗️ Build & Run

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/ecommerce-analytics-docker.git
   cd ecommerce-analytics-docker

Future Upgrades
### 1. Real-Time Data Streaming
- Integrate **Apache Kafka** or **AWS Kinesis** to handle live streams of sales and review data.
- Power dashboards with real-time updates—no manual refresh required.

### 2. Cloud-Native Deployment
- Migrate from local Docker to **AWS ECS/Fargate** or **Google Cloud Run**.
- Enable auto-scaling and high availability.
- Add **load balancing** for better performance under traffic spikes.

### 3. Serverless ML APIs
- Deploy **sentiment analysis** and **recommendation models** as **microservices** using **AWS Lambda** or **FastAPI**.
- Cut costs by using a **pay-per-inference** pricing model.

### 4. Time-Series Database
- Replace **SQLite** with **InfluxDB** or **TimescaleDB** for handling time-based sales analytics.
- Optimize long-term trend analysis and high-frequency queries.

### 5. Edge Caching
- Use **Cloudflare CDN** or similar edge services to globally cache static dashboard assets.
- Significantly reduce latency for international users and mobile devices.

---
