import streamlit as st
from supabase import create_client, Client
import os

def init_supabase():
    """Initialize Supabase client"""
    supabase_url = st.secrets["SUPABASE_URL"]
    supabase_key = st.secrets["SUPABASE_ANON_KEY"]
    return create_client(supabase_url, supabase_key)

def init_auth():
    """Initialize authentication"""
    if 'supabase' not in st.session_state:
        st.session_state.supabase = init_supabase()
    
    if 'user' not in st.session_state:
        st.session_state.user = None

def login():
    """Show login form"""
    with st.form("login"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        
        if submit:
            try:
                response = st.session_state.supabase.auth.sign_in_with_password({
                    "email": email,
                    "password": password
                })
                st.session_state.user = response.user
                st.success("Logged in successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Login failed: {str(e)}")

def signup():
    """Show signup form"""
    with st.form("signup"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Sign Up")
        
        if submit:
            try:
                response = st.session_state.supabase.auth.sign_up({
                    "email": email,
                    "password": password
                })
                st.success("Signup successful! Please check your email to verify your account.")
            except Exception as e:
                st.error(f"Signup failed: {str(e)}")

def logout():
    """Handle logout"""
    if st.session_state.user:
        st.session_state.supabase.auth.sign_out()
        st.session_state.user = None
        st.rerun() 