import streamlit as st
import pandas as pd
import os
import sys

# Add project root to path to import src modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.predict import predict_ticket

st.set_page_config(page_title="Smart Support Ticket Classifier", page_icon="🎫", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #fafafa; }
    .stButton>button { background-color: #4CAF50; color: white; border-radius: 8px; padding: 10px 24px; font-weight: bold; border: none; width: 100%; }
    .stButton>button:hover { background-color: #45a049; color: white; }
    .metric-card { text-align: center; padding: 20px; background-color: #262730; border-radius: 10px; border-left: 5px solid #4CAF50; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 20px; }
    h1, h2, h3, h4, h5, h6 { color: #fafafa !important; }
    .stAlert { background-color: #262730; color: #fafafa; }
    </style>
""", unsafe_allow_html=True)

def init_session_state() -> None:
    if 'history' not in st.session_state:
        st.session_state.history = []

def main() -> None:
    init_session_state()
    
    st.sidebar.title("🎫 Support Ticket AI")
    st.sidebar.markdown("---")
    st.sidebar.info("This AI-powered system automatically classifies customer support tickets into categories and assigns priority levels using Machine Learning and NLP.")
    
    st.title("🤖 Smart Support Ticket Classification & Priority Prediction")
    st.markdown("### Enter a customer support ticket below to predict its category and priority.")
    
    ticket_text = st.text_area("Ticket Description", height=150, placeholder="e.g., My payment was deducted twice. Please refund me ASAP!", help="Paste the customer's support ticket text here.")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_btn = st.button("🚀 Predict Ticket", use_container_width=True)
        
    if predict_btn and ticket_text.strip():
        with st.spinner("Analyzing ticket..."):
            result = predict_ticket(ticket_text)
            
        st.session_state.history.append({
            'Ticket Text': ticket_text[:100] + "..." if len(ticket_text) > 100 else ticket_text,
            'Category': result['category'],
            'Priority': result['priority'],
            'Category Confidence': f"{result['category_confidence']:.1f}%",
            'Priority Confidence': f"{result['priority_confidence']:.1f}%"
        })
        
        st.markdown("---")
        st.subheader("📊 Prediction Results")
        
        col_cat, col_pri = st.columns(2)
        with col_cat:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Predicted Category</h3>
                <h1 style="color: #4CAF50;">{result['category']}</h1>
                <p>Confidence: {result['category_confidence']:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
            st.progress(result['category_confidence'] / 100.0)
            
        with col_pri:
            priority_color = "#ff4b4b" if result['priority'] == 'High' else "#ffa500" if result['priority'] == 'Medium' else "#4CAF50"
            st.markdown(f"""
            <div class="metric-card">
                <h3>Predicted Priority</h3>
                <h1 style="color: {priority_color};">{result['priority']}</h1>
                <p>Confidence: {result['priority_confidence']:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
            st.progress(result['priority_confidence'] / 100.0)
            
        st.markdown("### 📈 Prediction Probabilities")
        col_prob1, col_prob2 = st.columns(2)
        with col_prob1:
            st.markdown("**Category Probabilities**")
            df_cat_probs = pd.DataFrame.from_dict(result['category_probabilities'], orient='index', columns=['Probability'])
            st.bar_chart(df_cat_probs)
        with col_prob2:
            st.markdown("**Priority Probabilities**")
            df_pri_probs = pd.DataFrame.from_dict(result['priority_probabilities'], orient='index', columns=['Probability'])
            st.bar_chart(df_pri_probs)
            
    elif predict_btn and not ticket_text.strip():
        st.error("Please enter some ticket text before predicting.")
        
    if st.session_state.history:
        st.markdown("---")
        st.subheader("🕒 Recent Prediction History")
        df_history = pd.DataFrame(st.session_state.history)
        st.dataframe(df_history, use_container_width=True)
        csv = df_history.to_csv(index=False).encode('utf-8')
        st.download_button(label="📥 Download History as CSV", data=csv, file_name='ticket_prediction_history.csv', mime='text/csv')
        
    if st.sidebar.button("🔄 Reset App"):
        st.session_state.history = []
        st.rerun()

if __name__ == "__main__":
    main()