import streamlit as st
from supabase import create_client

def init_supabase():
    """Initialize Supabase client"""
    supabase_url = st.secrets["SUPABASE_URL"]
    supabase_key = st.secrets["SUPABASE_ANON_KEY"]
    return create_client(supabase_url, supabase_key)

def init_auth():
    """Initialize authentication"""
    if 'user' not in st.session_state:
        st.session_state.user = None
        
    if 'supabase' not in st.session_state:
        st.session_state.supabase = init_supabase()
    
    # Try to restore session
    if 'session' in st.session_state:
        try:
            st.session_state.user = st.session_state.session.user
        except:
            st.session_state.user = None
            del st.session_state.session

def login():
    """Show login form"""
    with st.form("login", clear_on_submit=True):
        st.markdown("##### Welcome back! 👋")
        email = st.text_input("📧 Email")
        password = st.text_input("🔒 Password", type="password")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            submit = st.form_submit_button("Login", use_container_width=True)
        with col2:
            st.markdown("Forgot password?")
        
        if submit:
            if not email or not password:
                st.error("Please fill in all fields")
                return
                
            try:
                with st.spinner("Logging in..."):
                    response = st.session_state.supabase.auth.sign_in_with_password({
                        "email": email,
                        "password": password
                    })
                    # Store entire session object
                    st.session_state.session = response.session
                    st.session_state.user = response.user
                st.success("🎉 Logged in successfully!")
                st.rerun()
            except Exception as e:
                st.error("❌ Login failed. Please check your credentials.")

def logout():
    """Handle logout"""
    if st.session_state.user:
        st.session_state.supabase.auth.sign_out()
        st.session_state.user = None
        if 'session' in st.session_state:
            del st.session_state.session
        st.rerun()

def show_auth_ui():
    """Show stylish authentication UI"""
    # Custom CSS
    st.markdown("""
        <style>
            .auth-container {
                max-width: 400px;
                margin: 0 auto;
                padding: 2rem;
                border-radius: 10px;
                background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            .auth-header {
                text-align: center;
                margin-bottom: 2rem;
            }
            .auth-header img {
                width: 80px;
                margin-bottom: 1rem;
            }
            .auth-title {
                color: #1a1a1a;
                font-size: 1.8rem;
                font-weight: 700;
                margin-bottom: 0.5rem;
            }
            .auth-subtitle {
                color: #666;
                font-size: 1rem;
            }
            .stTabs > div > div {
                background-color: transparent !important;
            }
        </style>
    """, unsafe_allow_html=True)

    # Auth container
    st.markdown("""
        <div class="auth-container">
            <div class="auth-header">
                <img src="https://www.svgrepo.com/show/530438/machine-learning.svg" alt="Logo">
                <div class="auth-title">Tax Classification Assistant</div>
                <div class="auth-subtitle">Streamline your tax categorization</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Create tabs for login/signup
    tab1, tab2 = st.tabs(["🔑 Login", "✨ Sign Up"])
    
    with tab1:
        login()
    with tab2:
        signup()

def signup():
    """Show signup form"""
    with st.form("signup", clear_on_submit=True):
        st.markdown("##### Create an account 🚀")
        email = st.text_input("📧 Email")
        password = st.text_input("🔒 Password", type="password")
        password_confirm = st.text_input("🔒 Confirm Password", type="password", key="signup_password")
        
        submit = st.form_submit_button("Sign Up", use_container_width=True)
        
        if submit:
            if not email or not password or not password_confirm:
                st.error("Please fill in all fields")
                return
                
            if password != password_confirm:
                st.error("Passwords do not match")
                return
                
            try:
                with st.spinner("Creating your account..."):
                    response = st.session_state.supabase.auth.sign_up({
                        "email": email,
                        "password": password
                    })
                st.success("✨ Account created successfully! Please check your email to verify your account.")
            except Exception as e:
                st.error("❌ Signup failed. Please try again.")
                if st.session_state.get('debug_mode'):
                    st.error(str(e)) 