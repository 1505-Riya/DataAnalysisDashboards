import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg before importing pyplot
from flask import Flask, render_template, request, jsonify
import pandas as pd
import sqlite3
import os
import matplotlib.pyplot as plt
import io
import base64   
from contextlib import closing
import squarify
import gc
from sqlite3 import Error
import threading
from io import BytesIO
from ml_recommender import PrimeRecommender, NetflixRecommender  # Update import
from textblob import TextBlob  # Add this import at the top

app = Flask(__name__)
DB_FILE = "data.db"

# Create a thread-local storage for database connections
local = threading.local()

# Initialize the recommenders
prime_recommender = PrimeRecommender()
netflix_recommender = NetflixRecommender()

def get_db_connection():
    try:
        if not hasattr(local, 'connection'):
            local.connection = sqlite3.connect(DB_FILE, check_same_thread=False)
            local.connection.row_factory = sqlite3.Row
        return local.connection
    except sqlite3.Error as e:
        print(f"Database connection error: {e}")
        return None

def close_db_connection():
    try:
        if hasattr(local, 'connection'):
            local.connection.close()
            del local.connection
    except Exception as e:
        print(f"Error closing database connection: {e}")

@app.teardown_appcontext
def teardown_db(exception):
    close_db_connection()

def close_plot():
    try:
        plt.clf()
        plt.close('all')
        gc.collect()
    except Exception as e:
        print(f"Error in close_plot: {e}")

def create_plot():
    try:
        plt.clf()
        plt.close('all')
        fig = plt.figure(figsize=(10, 5), dpi=100)
        return fig
    except Exception as e:
        print(f"Error in create_plot: {e}")
        return None

def save_plot_to_base64():
    try:
        img = io.BytesIO()
        plt.savefig(img, format="png", bbox_inches="tight", dpi=100)
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        return plot_url
    except Exception as e:
        print(f"Error saving plot: {e}")
        return None
    finally:
        close_plot()

def handle_plot_error(e, route_name):
    print(f"Error in {route_name} route: {e}")
    close_plot()
    return "An error occurred while processing your request", 500

# Add a function to safely create and manage plots
def create_and_save_plot(plot_func, *args, **kwargs):
    try:
        fig = create_plot()
        if fig is None:
            return None
        plot_func(*args, **kwargs)
        return save_plot_to_base64()
    except Exception as e:
        print(f"Error in plot creation: {e}")
        return None
    finally:
        close_plot()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/head")
def head():
    try:
        with closing(get_db_connection()) as conn:
            if conn is None:
                return "Database connection error", 500
            df = pd.read_sql_query("SELECT * FROM merged_data LIMIT 10", conn)
            head_html = df.to_html(classes="table table-striped", index=False)
            return render_template("index.html", head_html=head_html)
    except Exception as e:
        return handle_plot_error(e, "head")

@app.route("/shape")
def shape():
    try:
        with closing(get_db_connection()) as conn:
            if conn is None:
                return "Database connection error", 500
            df = pd.read_sql_query("SELECT * FROM merged_data", conn)
            shape_html = f"<p><strong>Rows:</strong> {df.shape[0]}, <strong>Columns:</strong> {df.shape[1]}</p>"
            return render_template("index.html", shape_html=shape_html)
    except Exception as e:
        return handle_plot_error(e, "shape")

@app.route("/payment_mode_percentage")
def payment_mode_percentage():
    try:
        with closing(get_db_connection()) as conn:
            if conn is None:
                return "Database connection error", 500
            df = pd.read_sql_query("SELECT payment_mode FROM merged_data", conn)
            
            payment_counts = df["payment_mode"].value_counts(normalize=True) * 100
            
            def create_payment_plot():
                wedges, texts, autotexts = plt.pie(
                    payment_counts, 
                    labels=payment_counts.index, 
                    autopct="%1.1f%%", 
                    colors=plt.cm.Paired.colors
                )
                centre_circle = plt.Circle((0, 0), 0.70, fc="white")
                plt.gca().add_artist(centre_circle)
                plt.title("Payment Mode Distribution")
            
            plot_url = create_and_save_plot(create_payment_plot)
            if plot_url is None:
                return "Error creating plot", 500
            
            return render_template("index.html", plot_url=plot_url)
    except Exception as e:
        return handle_plot_error(e, "payment_mode_percentage")

@app.route("/profit_by_category")
def profit_by_category():
    try:
        with closing(get_db_connection()) as conn:
            if conn is None:
                return "Database connection error", 500
            df = pd.read_sql_query("SELECT category, SUM(profit) as total_profit FROM merged_data GROUP BY category", conn)
            
            def create_profit_plot():
                plt.bar(df["category"], df["total_profit"], color="lightcoral")
                plt.xlabel("Category")
                plt.ylabel("Total Profit")
                plt.title("Profit by Category")
                plt.xticks(rotation=45)
            
            plot_url = create_and_save_plot(create_profit_plot)
            if plot_url is None:
                return "Error creating plot", 500
            
            return render_template("index.html", plot_url=plot_url)
    except Exception as e:
        return handle_plot_error(e, "profit_by_category")

@app.route("/monthly_trend")
def monthly_trend():
    try:
        with closing(get_db_connection()) as conn:
            if conn is None:
                return "Database connection error", 500
            # First, let's check the date format in the database
            df = pd.read_sql_query("""
                SELECT order_date, amount 
                FROM merged_data 
                WHERE order_date IS NOT NULL 
                LIMIT 5
            """, conn)
            
            if df.empty:
                return "No data found", 404
                
            # Convert order_date to datetime if it's not already
            try:
                df['order_date'] = pd.to_datetime(df['order_date'])
            except Exception as e:
                print(f"Error converting dates: {str(e)}")
                # Try different date formats
                try:
                    df['order_date'] = pd.to_datetime(df['order_date'], format='%d-%m-%Y')
                except:
                    try:
                        df['order_date'] = pd.to_datetime(df['order_date'], format='%Y-%m-%d')
                    except:
                        return "Error: Unable to parse dates in the database", 500
            
            # Now get the monthly data
            df = pd.read_sql_query("""
                SELECT 
                    order_date,
                    amount
                FROM merged_data
                WHERE order_date IS NOT NULL
            """, conn)
            
            # Convert dates and aggregate
            df['order_date'] = pd.to_datetime(df['order_date'])
            df['month'] = df['order_date'].dt.strftime('%Y-%m')
            monthly_data = df.groupby('month')['amount'].sum().reset_index()
            monthly_data = monthly_data.sort_values('month')
            
            def create_trend_plot():
                plt.figure(figsize=(12, 6))
                plt.plot(monthly_data["month"], monthly_data["amount"], marker='o', color='teal', linewidth=2)
                plt.fill_between(monthly_data["month"], monthly_data["amount"], color='teal', alpha=0.2)
                plt.title('Monthly Sales Trend')
                plt.xlabel('Month')
                plt.ylabel('Total Sales ($)')
                plt.xticks(rotation=45)
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()
            
            plot_url = create_and_save_plot(create_trend_plot)
            if plot_url is None:
                return "Error creating plot", 500
            
            return render_template("index.html", plot_url=plot_url)
    except Exception as e:
        print(f"Error in monthly_trend: {str(e)}")
        return handle_plot_error(e, "monthly_trend")

@app.route("/monthly_orders")
def monthly_orders():
    try:
        with closing(get_db_connection()) as conn:
            if conn is None:
                return "Database connection error", 500
            # First, let's check the date format in the database
            df = pd.read_sql_query("""
                SELECT order_date, amount 
                FROM merged_data 
                WHERE order_date IS NOT NULL 
                LIMIT 5
            """, conn)
            
            if df.empty:
                return "No data found", 404
                
            # Convert order_date to datetime if it's not already
            try:
                df['order_date'] = pd.to_datetime(df['order_date'])
            except Exception as e:
                print(f"Error converting dates: {str(e)}")
                # Try different date formats
                try:
                    df['order_date'] = pd.to_datetime(df['order_date'], format='%d-%m-%Y')
                except:
                    try:
                        df['order_date'] = pd.to_datetime(df['order_date'], format='%Y-%m-%d')
                    except:
                        return "Error: Unable to parse dates in the database", 500
            
            # Now get the monthly data
            df = pd.read_sql_query("""
                SELECT 
                    order_date,
                    amount
                FROM merged_data
                WHERE order_date IS NOT NULL
            """, conn)
            
            # Convert dates and aggregate
            df['order_date'] = pd.to_datetime(df['order_date'])
            df['month'] = df['order_date'].dt.strftime('%Y-%m')
            monthly_data = df.groupby('month').size().reset_index(name='order_count')
            monthly_data = monthly_data.sort_values('month')
            
            def create_orders_plot():
                plt.figure(figsize=(12, 6))
                plt.plot(monthly_data["month"], monthly_data["order_count"], marker='o', color='slateblue', linewidth=2)
                plt.fill_between(monthly_data["month"], monthly_data["order_count"], color='slateblue', alpha=0.2)
                plt.title('Monthly Order Volume')
                plt.xlabel('Month')
                plt.ylabel('Number of Orders')
                plt.xticks(rotation=45)
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()
            
            plot_url = create_and_save_plot(create_orders_plot)
            if plot_url is None:
                return "Error creating plot", 500
            
            return render_template("index.html", plot_url=plot_url)
    except Exception as e:
        print(f"Error in monthly_orders: {str(e)}")
        return handle_plot_error(e, "monthly_orders")

@app.route("/subcategory_treemap")
def subcategory_treemap():
    try:
        with closing(get_db_connection()) as conn:
            if conn is None:
                return "Database connection error", 500
            df = pd.read_sql_query("""
                SELECT 
                    sub_category,
                    SUM(amount) as total_sales
                FROM merged_data
                GROUP BY sub_category
                ORDER BY total_sales DESC
            """, conn)
            
            # Calculate total sales and percentages
            total_sales = df['total_sales'].sum()
            df['percentage'] = (df['total_sales'] / total_sales * 100).round(1)
            
            # Create labels with percentages
            labels = [f"{row['sub_category']}\n({row['percentage']}%)" for _, row in df.iterrows()]
            
            def create_treemap():
                plt.figure(figsize=(12, 8))
                squarify.plot(
                    sizes=df["total_sales"],
                    label=labels,
                    color=plt.cm.Pastel1(range(len(df))),
                    alpha=0.7,
                    pad=True
                )
                plt.title('Sales Distribution by Sub-Category (with Percentages)')
                plt.axis('off')
            
            plot_url = create_and_save_plot(create_treemap)
            if plot_url is None:
                return "Error creating plot", 500
            
            return render_template("index.html", plot_url=plot_url)
    except Exception as e:
        return handle_plot_error(e, "subcategory_treemap")

@app.route("/top_products")
def top_products():
    try:
        with closing(get_db_connection()) as conn:
            if conn is None:
                return "Database connection error", 500
            try:
                df = pd.read_sql_query("""
                    SELECT 
                        sub_category as product_name,
                        COUNT(*) as order_count,
                        SUM(amount) as total_sales
                    FROM merged_data
                    GROUP BY sub_category
                    ORDER BY total_sales DESC
                    LIMIT 10
                """, conn)
                
                if df.empty:
                    return "No product data found", 404
                
                def create_top_products_plot():
                    plt.figure(figsize=(12, 6))
                    bars = plt.barh(df["product_name"], df["total_sales"], color='skyblue')
                    plt.title('Top 10 Products by Sales')
                    plt.xlabel('Total Sales')
                    plt.ylabel('Product Name')
                    
                    for bar in bars:
                        width = bar.get_width()
                        plt.text(width, bar.get_y() + bar.get_height()/2, 
                                f'₹{width:,.2f}', 
                                ha='left', va='center')
                    
                    plt.tight_layout()
                
                plot_url = create_and_save_plot(create_top_products_plot)
                if plot_url is None:
                    return "Error creating plot", 500
                
                return render_template("index.html", plot_url=plot_url)
            except sqlite3.OperationalError as e:
                print(f"SQL Error in top_products: {e}")
                return "Error accessing database", 500
    except Exception as e:
        return handle_plot_error(e, "top_products")

@app.route("/customer_segments")
def customer_segments():
    try:
        with closing(get_db_connection()) as conn:
            if conn is None:
                return "Database connection error", 500
            try:
                df = pd.read_sql_query("""
                    SELECT 
                        CASE 
                            WHEN total_spent < 1000 THEN 'Low Value'
                            WHEN total_spent < 5000 THEN 'Medium Value'
                            ELSE 'High Value'
                        END as segment,
                        COUNT(*) as customer_count,
                        AVG(total_spent) as avg_spent
                    FROM (
                        SELECT 
                            customer_name,
                            SUM(amount) as total_spent
                        FROM merged_data
                        GROUP BY customer_name
                    )
                    GROUP BY segment
                    ORDER BY avg_spent DESC
                """, conn)
                
                if df.empty:
                    return "No customer data found", 404
                
                def create_customer_segments_plot():
                    plt.figure(figsize=(8, 8))
                    wedges, texts, autotexts = plt.pie(
                        df["customer_count"],
                        labels=df["segment"],
                        autopct='%1.1f%%',
                        colors=plt.cm.Pastel1(range(len(df))),
                        startangle=90
                    )
                    
                    centre_circle = plt.Circle((0,0), 0.70, fc='white')
                    plt.gca().add_artist(centre_circle)
                    
                    plt.title('Customer Segments Distribution')
                    plt.axis('equal')
                
                plot_url = create_and_save_plot(create_customer_segments_plot)
                if plot_url is None:
                    return "Error creating plot", 500
                
                return render_template("index.html", plot_url=plot_url)
            except sqlite3.OperationalError as e:
                print(f"SQL Error in customer_segments: {e}")
                return "Error accessing database", 500
    except Exception as e:
        return handle_plot_error(e, "customer_segments")

@app.route("/avg_order_value")
def avg_order_value():
    try:
        with closing(get_db_connection()) as conn:
            if conn is None:
                return "Database connection error", 500
            try:
                df = pd.read_sql_query("""
                    SELECT 
                        substr(order_date, 7, 4) || '-' || substr(order_date, 4, 2) as month,
                        AVG(amount) as avg_order_value
                    FROM merged_data
                    WHERE order_date IS NOT NULL
                    GROUP BY month
                    ORDER BY month
                """, conn)
                
                if df.empty:
                    return "No order data found", 404
                
                def create_avg_order_plot():
                    plt.figure(figsize=(12, 6))
                    plt.plot(df["month"], df["avg_order_value"], marker='o', color='teal', linewidth=2)
                    plt.fill_between(df["month"], df["avg_order_value"], color='teal', alpha=0.2)
                    
                    plt.title('Average Order Value Trend')
                    plt.xlabel('Month')
                    plt.ylabel('Average Order Value (₹)')
                    plt.xticks(rotation=45)
                    plt.grid(True, linestyle='--', alpha=0.7)
                
                plot_url = create_and_save_plot(create_avg_order_plot)
                if plot_url is None:
                    return "Error creating plot", 500
                
                return render_template("index.html", plot_url=plot_url)
            except sqlite3.OperationalError as e:
                print(f"SQL Error in avg_order_value: {e}")
                return "Error accessing database", 500
    except Exception as e:
        return handle_plot_error(e, "avg_order_value")

@app.route("/regional_sales")
def regional_sales():
    try:
        with closing(get_db_connection()) as conn:
            if conn is None:
                return "Database connection error", 500
            try:
                df = pd.read_sql_query("""
                    SELECT 
                        state as region,
                        SUM(amount) as total_sales,
                        COUNT(*) as order_count
                    FROM merged_data
                    GROUP BY state
                    ORDER BY total_sales DESC
                """, conn)
                
                if df.empty:
                    return "No regional data found", 404
                
                def create_regional_sales_plot():
                    plt.figure(figsize=(12, 6))
                    x = range(len(df))
                    width = 0.35
                    
                    plt.bar(x, df["total_sales"], width, label='Total Sales', color='skyblue')
                    plt.bar([i + width for i in x], df["order_count"], width, label='Order Count', color='lightcoral')
                    
                    plt.title('Regional Sales Performance')
                    plt.xlabel('Region')
                    plt.ylabel('Amount')
                    plt.xticks([i + width/2 for i in x], df["region"], rotation=45)
                    plt.legend()
                    plt.grid(True, linestyle='--', alpha=0.7)
                
                plot_url = create_and_save_plot(create_regional_sales_plot)
                if plot_url is None:
                    return "Error creating plot", 500
                
                return render_template("index.html", plot_url=plot_url)
            except sqlite3.OperationalError as e:
                print(f"SQL Error in regional_sales: {e}")
                return "Error accessing database", 500
    except Exception as e:
        return handle_plot_error(e, "regional_sales")

@app.route("/payment_analysis")
def payment_analysis():
    try:
        with closing(get_db_connection()) as conn:
            if conn is None:
                return "Database connection error", 500
            try:
                df = pd.read_sql_query("""
                    SELECT 
                        payment_mode,
                        AVG(amount) as avg_order_value,
                        COUNT(*) as order_count
                    FROM merged_data
                    GROUP BY payment_mode
                    ORDER BY avg_order_value DESC
                """, conn)
                
                if df.empty:
                    return "No payment data found", 404
                
                def create_payment_analysis_plot():
                    plt.figure(figsize=(10, 6))
                    scatter = plt.scatter(
                        df["order_count"],
                        df["avg_order_value"],
                        s=df["avg_order_value"]*10,
                        c=range(len(df)),
                        cmap='viridis',
                        alpha=0.6
                    )
                    
                    for i, txt in enumerate(df["payment_mode"]):
                        plt.annotate(
                            txt,
                            (df["order_count"].iloc[i], df["avg_order_value"].iloc[i]),
                            xytext=(5, 5),
                            textcoords='offset points'
                        )
                    
                    plt.title('Payment Method Analysis')
                    plt.xlabel('Number of Orders')
                    plt.ylabel('Average Order Value (₹)')
                    plt.grid(True, linestyle='--', alpha=0.7)
                
                plot_url = create_and_save_plot(create_payment_analysis_plot)
                if plot_url is None:
                    return "Error creating plot", 500
                
                return render_template("index.html", plot_url=plot_url)
            except sqlite3.OperationalError as e:
                print(f"SQL Error in payment_analysis: {e}")
                return "Error accessing database", 500
    except Exception as e:
        return handle_plot_error(e, "payment_analysis")

def create_content_type_plot(df):
    try:
        if df.empty:
            return None
            
        plt.figure(figsize=(12, 6))
        bars = plt.bar(df['type'], df['count'], color='#E50914')
        
        # Add percentage labels
        total = df['count'].sum()
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height}\n({(height/total*100):.1f}%)',
                    ha='center', va='bottom')
        
        plt.title('Netflix Content Type Distribution', pad=20)
        plt.xlabel('Content Type')
        plt.ylabel('Number of Titles')
        plt.tight_layout()
        
        # Save plot to bytes
        img = BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        plt.close()
        
        return base64.b64encode(img.getvalue()).decode()
    except Exception as e:
        print(f"Error creating content type plot: {str(e)}")
        return None

def create_top_countries_plot(df):
    try:
        if df.empty:
            return None
            
        # Get top 10 countries by content count
        country_counts = df.set_index('country')['count'].sort_values(ascending=False).head(10)
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(country_counts.index, country_counts.values, color='#E50914')
        
        # Add percentage labels
        total = country_counts.sum()
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height}\n({(height/total*100):.1f}%)',
                    ha='center', va='bottom')
        
        plt.title('Top 10 Countries by Netflix Content', pad=20)
        plt.xlabel('Country')
        plt.ylabel('Number of Titles')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save plot to bytes
        img = BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        plt.close()
        
        return base64.b64encode(img.getvalue()).decode()
    except Exception as e:
        print(f"Error creating top countries plot: {str(e)}")
        return None

def create_ratings_plot(df):
    try:
        if df.empty:
            return None
            
        # Get rating distribution
        rating_counts = df.set_index('rating')['count'].sort_values(ascending=False)
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(rating_counts.index, rating_counts.values, color='#E50914')
        
        # Add percentage labels
        total = rating_counts.sum()
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height}\n({(height/total*100):.1f}%)',
                    ha='center', va='bottom')
        
        plt.title('Content Rating Distribution on Netflix', pad=20)
        plt.xlabel('Rating')
        plt.ylabel('Number of Titles')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save plot to bytes
        img = BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        plt.close()
        
        return base64.b64encode(img.getvalue()).decode()
    except Exception as e:
        print(f"Error creating ratings plot: {str(e)}")
        return None

def create_release_years_plot(df):
    try:
        if df.empty:
            return None
            
        # Get release year distribution
        year_counts = df.set_index('release_year')['count'].sort_index()
        
        plt.figure(figsize=(12, 6))
        plt.plot(year_counts.index, year_counts.values, color='#E50914', linewidth=2)
        plt.fill_between(year_counts.index, year_counts.values, color='#E50914', alpha=0.2)
        
        plt.title('Netflix Content Release Year Trends', pad=20)
        plt.xlabel('Release Year')
        plt.ylabel('Number of Titles')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot to bytes
        img = BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        plt.close()
        
        return base64.b64encode(img.getvalue()).decode()
    except Exception as e:
        print(f"Error creating release years plot: {str(e)}")
        return None

def create_top_genres_plot(df):
    try:
        if df.empty:
            return None
            
        # Split genres and count occurrences
        all_genres = []
        for genres in df['listed_in'].dropna():
            all_genres.extend([g.strip() for g in genres.split(',')])
        
        genre_counts = pd.Series(all_genres).value_counts().head(10)
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(genre_counts.index, genre_counts.values, color='#E50914')
        
        # Add percentage labels
        total = genre_counts.sum()
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height}\n({(height/total*100):.1f}%)',
                    ha='center', va='bottom')
        
        plt.title('Top 10 Genres on Netflix', pad=20)
        plt.xlabel('Genre')
        plt.ylabel('Number of Titles')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save plot to bytes
        img = BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        plt.close()
        
        return base64.b64encode(img.getvalue()).decode()
    except Exception as e:
        print(f"Error creating top genres plot: {str(e)}")
        return None

@app.route('/dashboard2', methods=['GET', 'POST'])
def dashboard2():
    try:
        # Handle recommendations form submission
        if request.method == 'POST':
            title = request.form.get('title')
            if title:
                recommendations = netflix_recommender.get_recommendations(title)
                return render_template('dashboard2.html', 
                                    recommendations=recommendations,
                                    plot_url=None)

        # Check if it's an AJAX request
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            query_type = request.args.get('query', 'content_type')
            
            # Connect to database
            conn = sqlite3.connect(DB_FILE)
            cursor = conn.cursor()
            
            # Get data based on query type
            if query_type == 'content_type':
                cursor.execute('''
                    SELECT type, COUNT(*) as count 
                    FROM netflix_titles 
                    GROUP BY type
                ''')
                df = pd.DataFrame(cursor.fetchall(), columns=['type', 'count'])
                plot_url = create_content_type_plot(df)
            elif query_type == 'top_countries':
                cursor.execute('''
                    SELECT country, COUNT(*) as count 
                    FROM netflix_titles 
                    WHERE country IS NOT NULL AND country != ''
                    GROUP BY country
                    ORDER BY count DESC
                    LIMIT 10
                ''')
                df = pd.DataFrame(cursor.fetchall(), columns=['country', 'count'])
                plot_url = create_top_countries_plot(df)
            elif query_type == 'ratings':
                cursor.execute('''
                    SELECT rating, COUNT(*) as count 
                    FROM netflix_titles 
                    WHERE rating IS NOT NULL AND rating != ''
                    GROUP BY rating
                    ORDER BY count DESC
                ''')
                df = pd.DataFrame(cursor.fetchall(), columns=['rating', 'count'])
                plot_url = create_ratings_plot(df)
            elif query_type == 'release_years':
                cursor.execute('''
                    SELECT release_year, COUNT(*) as count 
                    FROM netflix_titles 
                    WHERE release_year IS NOT NULL
                    GROUP BY release_year
                    ORDER BY release_year
                ''')
                df = pd.DataFrame(cursor.fetchall(), columns=['release_year', 'count'])
                plot_url = create_release_years_plot(df)
            elif query_type == 'top_genres':
                cursor.execute('''
                    SELECT listed_in 
                    FROM netflix_titles 
                    WHERE listed_in IS NOT NULL AND listed_in != ''
                ''')
                df = pd.DataFrame(cursor.fetchall(), columns=['listed_in'])
                plot_url = create_top_genres_plot(df)
            elif query_type == 'dataset_shape':
                cursor.execute('SELECT COUNT(*) FROM netflix_titles')
                total_rows = cursor.fetchone()[0]
                cursor.execute('PRAGMA table_info(netflix_titles)')
                total_columns = len(cursor.fetchall())
                shape_info = f"<p><strong>Total Rows:</strong> {total_rows:,}<br><strong>Total Columns:</strong> {total_columns}</p>"
                return render_template('visualization.html', shape_info=shape_info)
            elif query_type == 'top_rows':
                cursor.execute('SELECT * FROM netflix_titles LIMIT 20')
                df = pd.DataFrame(cursor.fetchall(), columns=[description[0] for description in cursor.description])
                table_html = df.to_html(classes='table table-striped table-hover', index=False)
                return render_template('visualization.html', table_html=table_html)
            
            conn.close()
            
            if plot_url:
                return render_template('visualization.html', plot_url=plot_url)
            else:
                return render_template('visualization.html', error="Error creating visualization")
        else:
            # Regular page load - show popular movies
            popular_movies = netflix_recommender.get_popular_movies()
            return render_template('dashboard2.html', recommendations=popular_movies)
    except Exception as e:
        print(f"Error in dashboard2: {str(e)}")
        return render_template('visualization.html', error=f"Error: {str(e)}")

def create_prime_content_type_plot(df):
    try:
        if df.empty:
            return None
            
        plt.figure(figsize=(12, 6))
        bars = plt.bar(df['type'], df['count'], color='#00A8E1')
        
        # Add percentage labels
        total = df['count'].sum()
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height}\n({(height/total*100):.1f}%)',
                    ha='center', va='bottom')
        
        plt.title('Amazon Prime Content Type Distribution', pad=20)
        plt.xlabel('Content Type')
        plt.ylabel('Number of Titles')
        plt.tight_layout()
        
        # Save plot to bytes
        img = BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        plt.close()
        
        return base64.b64encode(img.getvalue()).decode()
    except Exception as e:
        print(f"Error creating content type plot: {str(e)}")
        return None

def create_prime_top_countries_plot(df):
    try:
        if df.empty:
            return None
            
        # Get top 10 countries by content count
        country_counts = df.set_index('country')['count'].sort_values(ascending=False).head(10)
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(country_counts.index, country_counts.values, color='#00A8E1')
        
        # Add percentage labels
        total = country_counts.sum()
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height}\n({(height/total*100):.1f}%)',
                    ha='center', va='bottom')
        
        plt.title('Top 10 Countries by Amazon Prime Content', pad=20)
        plt.xlabel('Country')
        plt.ylabel('Number of Titles')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save plot to bytes
        img = BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        plt.close()
        
        return base64.b64encode(img.getvalue()).decode()
    except Exception as e:
        print(f"Error creating top countries plot: {str(e)}")
        return None

def create_prime_ratings_plot(df):
    try:
        if df.empty:
            return None
            
        # Get rating distribution
        rating_counts = df.set_index('rating')['count'].sort_values(ascending=False)
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(rating_counts.index, rating_counts.values, color='#00A8E1')
        
        # Add percentage labels
        total = rating_counts.sum()
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height}\n({(height/total*100):.1f}%)',
                    ha='center', va='bottom')
        
        plt.title('Content Rating Distribution on Amazon Prime', pad=20)
        plt.xlabel('Rating')
        plt.ylabel('Number of Titles')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save plot to bytes
        img = BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        plt.close()
        
        return base64.b64encode(img.getvalue()).decode()
    except Exception as e:
        print(f"Error creating ratings plot: {str(e)}")
        return None

def create_prime_release_years_plot(df):
    try:
        if df.empty:
            return None
            
        # Get release year distribution
        year_counts = df.set_index('release_year')['count'].sort_index()
        
        plt.figure(figsize=(12, 6))
        plt.plot(year_counts.index, year_counts.values, color='#00A8E1', linewidth=2)
        plt.fill_between(year_counts.index, year_counts.values, color='#00A8E1', alpha=0.2)
        
        plt.title('Amazon Prime Content Release Year Trends', pad=20)
        plt.xlabel('Release Year')
        plt.ylabel('Number of Titles')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot to bytes
        img = BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        plt.close()
        
        return base64.b64encode(img.getvalue()).decode()
    except Exception as e:
        print(f"Error creating release years plot: {str(e)}")
        return None

def create_prime_top_genres_plot(df):
    try:
        if df.empty:
            return None
            
        # Split genres and count occurrences
        all_genres = []
        for genres in df['listed_in'].dropna():
            all_genres.extend([g.strip() for g in genres.split(',')])
        
        genre_counts = pd.Series(all_genres).value_counts().head(10)
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(genre_counts.index, genre_counts.values, color='#00A8E1')
        
        # Add percentage labels
        total = genre_counts.sum()
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height}\n({(height/total*100):.1f}%)',
                    ha='center', va='bottom')
        
        plt.title('Top 10 Genres on Amazon Prime', pad=20)
        plt.xlabel('Genre')
        plt.ylabel('Number of Titles')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save plot to bytes
        img = BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        plt.close()
        
        return base64.b64encode(img.getvalue()).decode()
    except Exception as e:
        print(f"Error creating top genres plot: {str(e)}")
        return None

def analyze_sentiment(text):
    try:
        if not text:
            return None
            
        analysis = TextBlob(text)
        sentiment_score = analysis.sentiment.polarity
        subjectivity_score = analysis.sentiment.subjectivity
        
        # Determine sentiment label
        if sentiment_score > 0.1:
            sentiment = "Positive"
            color = "success"
        elif sentiment_score < -0.1:
            sentiment = "Negative"
            color = "danger"
        else:
            sentiment = "Neutral"
            color = "warning"
            
        # Determine subjectivity label
        if subjectivity_score > 0.6:
            subjectivity_text = "Highly Subjective"
        elif subjectivity_score > 0.3:
            subjectivity_text = "Moderately Subjective"
        else:
            subjectivity_text = "Objective"
            
        return {
            "sentiment": sentiment,
            "sentiment_score": sentiment_score,
            "color": color,
            "subjectivity": subjectivity_score,
            "subjectivity_text": subjectivity_text,
            "description": f"This content has a {sentiment.lower()} sentiment with a score of {sentiment_score:.2f}. The text is {subjectivity_text.lower()} with a subjectivity score of {subjectivity_score:.2f}."
        }
    except Exception as e:
        print(f"Error in sentiment analysis: {str(e)}")
        return None

@app.route('/dashboard3', methods=['GET', 'POST'])
def dashboard3():
    try:
        # Handle sentiment analysis form submission
        if request.method == 'POST' and request.form.get('analysis_type') == 'sentiment':
            title = request.form.get('title')
            if title:
                # Connect to database
                conn = sqlite3.connect('amazon_prime.db')
                cursor = conn.cursor()
                
                # Get description for the title
                cursor.execute('''
                    SELECT description 
                    FROM amazon_prime_titles 
                    WHERE title = ? 
                    LIMIT 1
                ''', (title,))
                
                result = cursor.fetchone()
                conn.close()
                
                if result:
                    description = result[0]
                    sentiment_analysis = analyze_sentiment(description)
                    if sentiment_analysis:
                        return render_template('dashboard3.html', 
                                            sentiment_analysis=sentiment_analysis,
                                            plot_url=None)
                
                return render_template('dashboard3.html', 
                                    error="Title not found or no description available",
                                    plot_url=None)

        # Handle recommendations form submission
        if request.method == 'POST':
            title = request.form.get('title')
            if title:
                recommendations = prime_recommender.get_recommendations(title)
                return render_template('dashboard3.html', 
                                    recommendations=recommendations,
                                    plot_url=None)

        # Check if it's an AJAX request
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            query_type = request.args.get('query', 'content_type')
            
            # Connect to database
            conn = sqlite3.connect('amazon_prime.db')
            cursor = conn.cursor()
            
            # Get data based on query type
            if query_type == 'content_type':
                cursor.execute('''
                    SELECT type, COUNT(*) as count 
                    FROM amazon_prime_titles 
                    GROUP BY type
                ''')
                df = pd.DataFrame(cursor.fetchall(), columns=['type', 'count'])
                plot_url = create_prime_content_type_plot(df)
            elif query_type == 'top_countries':
                cursor.execute('''
                    SELECT country, COUNT(*) as count 
                    FROM amazon_prime_titles 
                    WHERE country IS NOT NULL AND country != ''
                    GROUP BY country
                    ORDER BY count DESC
                    LIMIT 10
                ''')
                df = pd.DataFrame(cursor.fetchall(), columns=['country', 'count'])
                plot_url = create_prime_top_countries_plot(df)
            elif query_type == 'ratings':
                cursor.execute('''
                    SELECT rating, COUNT(*) as count 
                    FROM amazon_prime_titles 
                    WHERE rating IS NOT NULL AND rating != ''
                    GROUP BY rating
                    ORDER BY count DESC
                ''')
                df = pd.DataFrame(cursor.fetchall(), columns=['rating', 'count'])
                plot_url = create_prime_ratings_plot(df)
            elif query_type == 'release_years':
                cursor.execute('''
                    SELECT release_year, COUNT(*) as count 
                    FROM amazon_prime_titles 
                    WHERE release_year IS NOT NULL
                    GROUP BY release_year
                    ORDER BY release_year
                ''')
                df = pd.DataFrame(cursor.fetchall(), columns=['release_year', 'count'])
                plot_url = create_prime_release_years_plot(df)
            elif query_type == 'top_genres':
                cursor.execute('''
                    SELECT listed_in 
                    FROM amazon_prime_titles 
                    WHERE listed_in IS NOT NULL AND listed_in != ''
                ''')
                df = pd.DataFrame(cursor.fetchall(), columns=['listed_in'])
                plot_url = create_prime_top_genres_plot(df)
            elif query_type == 'dataset_shape':
                cursor.execute('SELECT COUNT(*) FROM amazon_prime_titles')
                total_rows = cursor.fetchone()[0]
                cursor.execute('PRAGMA table_info(amazon_prime_titles)')
                total_columns = len(cursor.fetchall())
                shape_info = f"<p><strong>Total Rows:</strong> {total_rows:,}<br><strong>Total Columns:</strong> {total_columns}</p>"
                return render_template('dashboard3.html', shape_info=shape_info)
            elif query_type == 'top_rows':
                cursor.execute('SELECT * FROM amazon_prime_titles LIMIT 20')
                df = pd.DataFrame(cursor.fetchall(), columns=[description[0] for description in cursor.description])
                table_html = df.to_html(classes='table table-striped table-hover', index=False)
                return render_template('dashboard3.html', table_html=table_html)
            
            conn.close()
            
            if plot_url:
                return render_template('dashboard3.html', plot_url=plot_url)
            else:
                return render_template('dashboard3.html', error="Error creating visualization")
        else:
            # Regular page load - show popular movies
            popular_movies = prime_recommender.get_popular_movies()
            return render_template('dashboard3.html', recommendations=popular_movies)
    except Exception as e:
        print(f"Error in dashboard3: {str(e)}")
        return render_template('dashboard3.html', error=f"Error: {str(e)}")

@app.route('/recommendations', methods=['GET', 'POST'])
def recommendations():
    if request.method == 'POST':
        title = request.form.get('title')
        if title:
            recommendations = prime_recommender.get_recommendations(title)
            return render_template('recommendations.html', 
                                 recommendations=recommendations,
                                 search_title=title)
    
    # Get popular movies for the initial view
    popular_movies = prime_recommender.get_popular_movies()
    return render_template('recommendations.html', 
                         recommendations=popular_movies,
                         search_title=None)

if __name__ == "__main__":
    app.run(debug=True, threaded=True)
