import streamlit as st
from auth import register_user, login_user
from oauth_handler import initiate_oauth_login, handle_oauth_callback

def social_login(provider):
    """Handle social login with OAuth"""
    initiate_oauth_login(provider)

def show_login_page():
    """Display professional login/signup page with social login options"""
    
    # Set page style FIRST (must be before any other Streamlit commands)
    st.set_page_config(
        page_title="Marine Debris Analytics - Secure Login",
        page_icon="🌊",
        layout="centered",
        initial_sidebar_state="collapsed"
    )
    
    # Check for OAuth callback and stop rendering if successful
    if handle_oauth_callback():
        st.success("✅ Successfully logged in with OAuth!")
        st.balloons()
        st.rerun()
        return  # Exit function to prevent duplicate rendering
    
    
    # Professional custom CSS with animations and modern design
    st.markdown("""
        <style>
        /* Advanced Ocean Color Palette */
        :root {
            --ocean-deep: #0a1929;
            --ocean-primary: #0284c7;
            --ocean-light: #0ea5e9;
            --ocean-cyan: #06b6d4;
            --ocean-teal: #14b8a6;
            --sand-light: #f0f9ff;
            --coral-accent: #f97316;
            --success-green: #10b981;
        }
        
        /* Full-page background (marine debris) */
        html, body, [data-testid="stAppViewContainer"] {
            background:
                linear-gradient(135deg,
                    rgba(10, 19, 33, 0.92) 0%,
                    rgba(15, 32, 52, 0.86) 35%,
                    rgba(6, 110, 164, 0.78) 100%),
                url('https://images.unsplash.com/photo-1559827260-dc66d52bef19?auto=format&fit=crop&w=2400&q=80') center/cover fixed !important;
            min-height: 100vh;
        }

        /* Main container with gentle motion */
        .main {
            background: transparent;
            position: relative;
            animation: bgShift 16s ease-in-out infinite alternate;
        }
        
        @keyframes bgShift {
            0% { background-position: center center; }
            100% { background-position: center bottom; }
        }
        
        /* Animated wave effect overlay */
        .main::before {
            content: '';
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            height: 250px;
            background: linear-gradient(180deg, 
                transparent 0%, 
                rgba(15, 23, 42, 0.5) 50%,
                rgba(15, 23, 42, 0.8) 100%
            );
            pointer-events: none;
            animation: waveMove 8s ease-in-out infinite;
        }
        
        @keyframes waveMove {
            0%, 100% { opacity: 0.6; }
            50% { opacity: 0.8; }
        }
        
        /* Hide default header bar */
        [data-testid="stHeader"] { background: transparent !important; }

        /* Bright text for hero/outside card areas */
        .main, .main p, .main span, .main small, .main label {
            color: #38bdf8;
        }

        /* Keep card/body text dark for contrast */
        .login-container, .login-container p, .login-container span, .login-container label, .login-container small {
            color: #0b1729;
        }

        /* Ensure background sizing */
        [data-testid="stAppViewContainer"] {
            background-size: cover !important;
            background-attachment: fixed !important;
        }
        
        /* Glassmorphism container */
        .block-container {
            background: rgba(255, 255, 255, 0.10) !important;
            backdrop-filter: blur(20px) saturate(180%);
            border: 1px solid rgba(255, 255, 255, 0.18);
            border-radius: 24px !important;
            box-shadow: 
                0 20px 60px rgba(0, 0, 0, 0.3),
                inset 0 1px 0 rgba(255, 255, 255, 0.1);
        }
        
        /* Stmetrics styling */
        [data-testid="stMetricContainer"] {
            background-color: transparent;
            padding: 0;
        }
        
        /* Enhanced login container with glassmorphism */
        .login-container {
            background: rgba(255, 255, 255, 0.97);
            border-radius: 20px;
            box-shadow: 
                0 16px 48px rgba(0, 0, 0, 0.25),
                0 0 80px rgba(14, 165, 233, 0.12);
            padding: 3rem;
            max-width: 480px;
            animation: slideUp 0.8s cubic-bezier(0.16, 1, 0.3, 1);
            border: 1px solid rgba(14, 165, 233, 0.22);
        }
        
        @keyframes slideUp {
            from {
                opacity: 0;
                transform: translateY(40px) scale(0.95);
            }
            to {
                opacity: 1;
                transform: translateY(0) scale(1);
            }
        }
        
        /* Header styling with gradient text */
        .header-container {
            text-align: center;
            margin-bottom: 2.5rem;
            position: relative;
        }
        
        /* Animated wave icon */
        .logo {
            font-size: 4rem;
            margin-bottom: 0.5rem;
            animation: float 3s ease-in-out infinite;
            filter: drop-shadow(0 4px 12px rgba(14, 165, 233, 0.3));
        }
        
        @keyframes float {
            0%, 100% { transform: translateY(0px) rotate(0deg); }
            25% { transform: translateY(-8px) rotate(-3deg); }
            75% { transform: translateY(-8px) rotate(3deg); }
        }
        
        .header-title {
            font-size: 2.3rem;
            font-weight: 850;
            background: linear-gradient(135deg, 
                #38bdf8 0%, 
                #7dd3fc 50%,
                #bae6fd 100%
            );
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 0.5rem;
            letter-spacing: -0.4px;
            text-shadow: 0 2px 12px rgba(56, 189, 248, 0.5);
        }
        
        .header-subtitle {
            color: #7dd3fc;
            font-size: 0.95rem;
            font-weight: 600;
            letter-spacing: 0.3px;
        }
        
        /* Modern tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
            background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%);
            border-radius: 12px;
            padding: 0.4rem;
            margin-bottom: 2rem;
            box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.06);
        }
        
        .stTabs [data-baseweb="tab"] {
            background-color: transparent;
            border-radius: 10px;
            font-weight: 600;
            color: #38bdf8;
            padding: 0.8rem 1.5rem;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background: linear-gradient(135deg, 
                var(--ocean-primary) 0%, 
                var(--ocean-light) 100%
            );
            color: white;
            box-shadow: 0 4px 12px rgba(14, 165, 233, 0.4);
            transform: translateY(-1px);
        }
        
        /* Enhanced input styling */
        .stTextInput > div > div > input {
            border: 2px solid #e2e8f0;
            border-radius: 10px;
            padding: 0.85rem 1.2rem;
            font-size: 1rem;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            background: white;
        }
        
        .stTextInput > div > div > input:focus {
            border-color: var(--ocean-light);
            box-shadow: 
                0 0 0 4px rgba(14, 165, 233, 0.1),
                0 4px 12px rgba(14, 165, 233, 0.15);
            transform: translateY(-1px);
        }
        
        /* Form label with icon support */
        .stTextInput > label {
            font-weight: 700;
            color: #38bdf8;
            font-size: 0.98rem;
            margin-bottom: 0.6rem;
            letter-spacing: 0.2px;
        }
        
        /* Premium button design */
        .stButton > button {
            background: linear-gradient(135deg, 
                var(--ocean-primary) 0%, 
                var(--ocean-light) 50%,
                var(--ocean-cyan) 100%
            );
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.85rem 2rem;
            font-weight: 700;
            font-size: 1rem;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 4px 16px rgba(14, 165, 233, 0.4);
            position: relative;
            overflow: hidden;
        }
        
        .stButton > button::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
            transition: left 0.5s;
        }
        
        .stButton > button:hover::before {
            left: 100%;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(14, 165, 233, 0.5);
        }
        
        .stButton > button:active {
            transform: translateY(0);
            box-shadow: 0 4px 12px rgba(14, 165, 233, 0.4);
        }
        
        /* Social login buttons with hover effects */
        .stButton > button[kind="secondary"] {
            background: white;
            color: #1f2937;
            border: 2px solid #dfe7f1;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.08);
            font-weight: 600;
            font-size: 0.95rem;
            padding: 0.9rem 1.3rem;
        }
        
        .stButton > button[kind="secondary"]:hover {
            background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
            border-color: var(--ocean-light);
            box-shadow: 0 6px 18px rgba(14, 165, 233, 0.22);
            transform: translateY(-2px);
        }
        
        /* Divider with gradient */
        .stDivider {
            margin: 1.5rem 0;
            border: none;
            height: 1px;
            background: linear-gradient(90deg, 
                transparent 0%, 
                #cbd5e1 50%, 
                transparent 100%
            );
        }
        
        /* Enhanced message styling */
        .stSuccess {
            background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
            border-radius: 12px;
            padding: 1rem 1.2rem;
            border-left: 4px solid var(--success-green);
            box-shadow: 0 4px 12px rgba(16, 185, 129, 0.1);
        }
        
        .stError {
            background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
            border-radius: 12px;
            padding: 1rem 1.2rem;
            border-left: 4px solid #ef4444;
            box-shadow: 0 4px 12px rgba(239, 68, 68, 0.1);
        }
        
        .stInfo {
            background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
            border-radius: 12px;
            padding: 1rem 1.2rem;
            border-left: 4px solid var(--ocean-light);
            box-shadow: 0 4px 12px rgba(14, 165, 233, 0.1);
        }
        
        /* Caption styling */
        .stCaption {
            color: #38bdf8;
            font-size: 0.92rem;
            font-weight: 600;
            text-align: center;
            margin-top: 1.5rem;
        }
        
        /* Enhanced footer */
        .footer {
            text-align: center;
            color: #94a3b8;
            font-size: 0.85rem;
            margin-top: 2rem;
            padding-top: 1.5rem;
            border-top: 1px solid #e2e8f0;
        }
        
        /* Feature cards with hover effects */
        .features {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.2rem;
            margin-top: 1.5rem;
            padding-top: 1.5rem;
            border-top: 1px solid #e2e8f0;
        }
        
        .feature-item {
            padding: 1rem;
            text-align: center;
            color: #38bdf8;
            font-size: 0.85rem;
            background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
            border-radius: 10px;
            transition: all 0.3s ease;
            border: 1px solid #e2e8f0;
        }
        
        .feature-item:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(14, 165, 233, 0.15);
            border-color: var(--ocean-light);
        }
        
        .feature-icon {
            font-size: 1.8rem;
            margin-bottom: 0.5rem;
            filter: drop-shadow(0 2px 4px rgba(0, 0, 0, 0.1));
        }
        
        /* Subheader with accent */
        .subheader {
            font-size: 1.22rem;
            font-weight: 750;
            color: #38bdf8;
            margin-bottom: 1.5rem;
            position: relative;
            padding-left: 1rem;
        }
        
        .subheader::before {
            content: '';
            position: absolute;
            left: 0;
            top: 50%;
            transform: translateY(-50%);
            width: 4px;
            height: 60%;
            background: linear-gradient(180deg, var(--ocean-primary), var(--ocean-cyan));
            border-radius: 2px;
        }
        
        /* Enhanced password strength indicator */
        .password-strength {
            height: 5px;
            background-color: #e2e8f0;
            border-radius: 3px;
            margin-top: 0.6rem;
            overflow: hidden;
            box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.1);
        }
        
        .password-strength-bar {
            height: 100%;
            border-radius: 3px;
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        }
        
        .password-weak { 
            background: linear-gradient(90deg, #ef4444, #f87171); 
            width: 33%; 
        }
        .password-medium { 
            background: linear-gradient(90deg, #f59e0b, #fbbf24); 
            width: 66%; 
        }
        .password-strong { 
            background: linear-gradient(90deg, var(--success-green), #34d399); 
            width: 100%; 
        }
        
        /* Checkbox styling */
        .stCheckbox {
            padding: 0.5rem 0;
        }
        
        /* Smooth scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: #f1f5f9;
        }
        
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(180deg, var(--ocean-primary), var(--ocean-light));
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: var(--ocean-cyan);
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Wrapper for centering
    col1, col2, col3 = st.columns([0.5, 1, 0.5])
    
    with col2:
        # Header
        st.markdown("""
            <div class="header-container">
                <div class="logo">🌊</div>
                <div class="header-title">Marine Debris</div>
                <div class="header-subtitle">Prediction & Risk Analysis Platform</div>
            </div>
        """, unsafe_allow_html=True)
        
        # Tab selection
        tab1, tab2 = st.tabs(["🔑 Login", "📝 Sign Up"])
        
        with tab1:
            st.markdown('<div class="subheader">Welcome Back</div>', unsafe_allow_html=True)
            st.markdown("Login to access your marine debris analytics dashboard")
            
            st.markdown("")  # Spacing
            
            username = st.text_input(
                "👤 Username",
                placeholder="Enter your username",
                key="login_username"
            )
            
            password = st.text_input(
                "🔐 Password",
                type="password",
                placeholder="Enter your password",
                key="login_password"
            )
            
            col_btn1, col_btn2 = st.columns([1, 1])
            with col_btn1:
                if st.button("🔓 Login", type="primary", use_container_width=True):
                    if username and password:
                        success, message = login_user(username, password)
                        if success:
                            st.session_state.logged_in = True
                            st.session_state.username = username
                            st.success(f"✅ {message}")
                            st.balloons()
                            st.rerun()
                        else:
                            st.error(f"❌ {message}")
                    else:
                        st.error("❌ Please enter both username and password")
            
            with col_btn2:
                if st.button("🔄 Reset", use_container_width=True):
                    st.info("Contact support for password reset")
            
            st.markdown("---")
            st.markdown('<div style="text-align: center; color: #666; font-size: 0.9rem; margin: 1rem 0;">Or continue with</div>', unsafe_allow_html=True)
            
            # Social Login Buttons
            col_g, col_gh, col_f = st.columns(3)
            with col_g:
                if st.button("🔵 Google", use_container_width=True, key="google_login", type="secondary"):
                    social_login("Google")
            with col_gh:
                if st.button("⚫ GitHub", use_container_width=True, key="github_login", type="secondary"):
                    social_login("GitHub")
            with col_f:
                if st.button("🔷 Facebook", use_container_width=True, key="facebook_login", type="secondary"):
                    social_login("Facebook")
            
            # Additional social options in a second row
            col_ms, col_tw, col_ap = st.columns(3)
            with col_ms:
                if st.button("🟦 Microsoft", use_container_width=True, key="microsoft_login", type="secondary"):
                    social_login("Microsoft")
            with col_tw:
                if st.button("🐦 Twitter/X", use_container_width=True, key="twitter_login", type="secondary"):
                    social_login("Twitter")
            with col_ap:
                if st.button("🍎 Apple", use_container_width=True, key="apple_login", type="secondary"):
                    social_login("Apple")
            
            st.markdown("---")
            
            st.markdown('<div style="text-align: center; color: #0b1729; font-size: 0.96rem; font-weight: 700; margin-top: 1rem;">Don\'t have an account? <br><strong>Sign up above 👆</strong></div>', unsafe_allow_html=True)
        
        with tab2:
            st.markdown('<div class="subheader">Create Your Account</div>', unsafe_allow_html=True)
            st.markdown("Join our platform to analyze marine debris patterns")
            
            st.markdown("")  # Spacing
            
            reg_username = st.text_input(
                "👤 Username",
                placeholder="At least 3 characters (letters, numbers, underscore)",
                key="reg_username",
                help="Choose a unique username for your account"
            )
            
            reg_email = st.text_input(
                "📧 Email Address",
                placeholder="your@email.com",
                key="reg_email",
                help="We'll use this for account recovery"
            )
            
            reg_password = st.text_input(
                "🔐 Password",
                type="password",
                placeholder="At least 6 characters (mix of letters & numbers)",
                key="reg_password",
                help="Create a strong password"
            )
            
            # Password strength indicator
            if reg_password:
                strength = "weak" if len(reg_password) < 8 else "medium" if len(reg_password) < 12 else "strong"
                strength_class = f"password-{strength}"
                strength_text = {"weak": "Weak", "medium": "Medium", "strong": "Strong"}
                st.markdown(f"""
                    <div class="password-strength">
                        <div class="password-strength-bar {strength_class}"></div>
                    </div>
                    <small style="color: #666;">Strength: {strength_text[strength]}</small>
                """, unsafe_allow_html=True)
            
            reg_password_confirm = st.text_input(
                "🔐 Confirm Password",
                type="password",
                placeholder="Confirm your password",
                key="reg_password_confirm"
            )
            
            # Terms checkbox
            agree_terms = st.checkbox(
                "I agree to the Terms of Service and Privacy Policy",
                value=False
            )
            
            col_create, col_cancel = st.columns([1, 1])
            with col_create:
                if st.button("✍️ Create Account", type="primary", use_container_width=True):
                    if not reg_username or not reg_email or not reg_password:
                        st.error("❌ Please fill in all fields")
                    elif reg_password != reg_password_confirm:
                        st.error("❌ Passwords do not match")
                    elif not agree_terms:
                        st.error("❌ Please agree to the terms and conditions")
                    else:
                        success, message = register_user(reg_username, reg_email, reg_password)
                        if success:
                            st.success(f"✅ {message}")
                            st.info("🎉 Welcome! Now go to the Login tab and enter your credentials")
                        else:
                            st.error(f"❌ {message}")
            
            with col_cancel:
                if st.button("❌ Clear", use_container_width=True):
                    st.rerun()
            
            st.markdown("---")
            
            st.markdown('<div style="text-align: center; color: #0b1729; font-size: 0.96rem; font-weight: 700; margin-top: 1rem;">Already have an account? <br><strong>Login above 👆</strong></div>', unsafe_allow_html=True)
        
        # Footer
        st.markdown("---")
        st.markdown("""
            <div class="footer">
                <div style="margin-bottom: 0.5rem;">
                    <strong>🌍 Marine Debris Prediction & Risk Mapping System</strong>
                </div>
                <div style="font-size: 0.8rem; color: #aaa; margin-bottom: 0.5rem;">
                    Version 2.0 • Powered by Advanced ML & GIS Analytics
                </div>
                <div style="font-size: 0.75rem; color: #aaa;">
                    © 2025 Marine Debris Analytics. All rights reserved. | 
                    <a href="#" style="color: #0ea5e9; text-decoration: none;">Privacy Policy</a> | 
                    <a href="#" style="color: #0ea5e9; text-decoration: none;">Terms of Service</a>
                </div>
            </div>
        """, unsafe_allow_html=True)

# Note: show_login_page() is called from app.py, not here
# This file only defines the function

