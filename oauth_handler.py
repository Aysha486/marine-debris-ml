"""
OAuth Authentication Handler for Social Login
Handles OAuth flow for multiple providers
"""

import streamlit as st
import requests
import secrets
import json
from urllib.parse import urlencode, parse_qs, urlparse
from oauth_config import OAUTH_PROVIDERS, get_provider_credentials
from auth import register_user

def generate_state():
    """Generate a random state parameter for OAuth security"""
    return secrets.token_urlsafe(32)

def build_auth_url(provider):
    """Build the OAuth authorization URL for a provider"""
    config = OAUTH_PROVIDERS.get(provider)
    if not config or not config['enabled']:
        return None
    
    client_id, _, redirect_uri = get_provider_credentials(provider)
    if not client_id:
        return None
    
    state = generate_state()
    st.session_state[f'oauth_state_{provider}'] = state
    st.session_state['oauth_provider'] = provider
    
    params = {
        'client_id': client_id,
        'redirect_uri': redirect_uri,
        'scope': ' '.join(config['scopes']),
        'response_type': 'code',
        'state': state
    }
    
    # Provider-specific parameters
    if provider == "Google":
        params['access_type'] = 'offline'
        params['prompt'] = 'consent'
    elif provider == "Microsoft":
        params['response_mode'] = 'query'
    
    return f"{config['auth_url']}?{urlencode(params)}"

def exchange_code_for_token(provider, code):
    """Exchange authorization code for access token"""
    config = OAUTH_PROVIDERS.get(provider)
    if not config:
        return None
    
    client_id, client_secret, redirect_uri = get_provider_credentials(provider)
    
    data = {
        'client_id': client_id,
        'client_secret': client_secret,
        'code': code,
        'redirect_uri': redirect_uri,
        'grant_type': 'authorization_code'
    }
    
    headers = {'Accept': 'application/json'}
    
    try:
        response = requests.post(config['token_url'], data=data, headers=headers)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Token exchange failed: {str(e)}")
        return None

def get_user_info(provider, access_token):
    """Fetch user information from OAuth provider"""
    config = OAUTH_PROVIDERS.get(provider)
    if not config or not config['user_info_url']:
        return None
    
    headers = {'Authorization': f'Bearer {access_token}'}
    
    # Provider-specific parameters
    if provider == "Facebook":
        params = {'fields': 'id,name,email'}
        response = requests.get(config['user_info_url'], headers=headers, params=params)
    else:
        response = requests.get(config['user_info_url'], headers=headers)
    
    try:
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Failed to get user info: {str(e)}")
        return None

def normalize_user_data(provider, user_data):
    """Normalize user data from different providers to a common format"""
    if provider == "Google":
        return {
            'id': user_data.get('id'),
            'email': user_data.get('email'),
            'name': user_data.get('name'),
            'username': user_data.get('email', '').split('@')[0]
        }
    elif provider == "GitHub":
        return {
            'id': user_data.get('id'),
            'email': user_data.get('email') or f"{user_data.get('login')}@github.local",
            'name': user_data.get('name') or user_data.get('login'),
            'username': user_data.get('login')
        }
    elif provider == "Facebook":
        return {
            'id': user_data.get('id'),
            'email': user_data.get('email'),
            'name': user_data.get('name'),
            'username': user_data.get('email', '').split('@')[0]
        }
    elif provider == "Microsoft":
        return {
            'id': user_data.get('id'),
            'email': user_data.get('mail') or user_data.get('userPrincipalName'),
            'name': user_data.get('displayName'),
            'username': user_data.get('userPrincipalName', '').split('@')[0]
        }
    elif provider == "Twitter":
        return {
            'id': user_data.get('data', {}).get('id'),
            'email': user_data.get('data', {}).get('username') + '@twitter.local',
            'name': user_data.get('data', {}).get('name'),
            'username': user_data.get('data', {}).get('username')
        }
    return None

def handle_oauth_callback():
    """Handle OAuth callback after user authorizes"""
    # Check for OAuth callback parameters in URL
    query_params = st.query_params
    
    if 'code' in query_params and 'state' in query_params:
        code = query_params['code']
        state = query_params['state']
        
        # Get provider from session
        provider = st.session_state.get('oauth_provider')
        if not provider:
            return False
        
        # Verify state to prevent CSRF
        expected_state = st.session_state.get(f'oauth_state_{provider}')
        if state != expected_state:
            st.error("⚠️ Security validation failed. Please try again.")
            return False
        
        # Exchange code for token
        token_data = exchange_code_for_token(provider, code)
        if not token_data:
            return False
        
        access_token = token_data.get('access_token')
        if not access_token:
            st.error("Failed to obtain access token")
            return False
        
        # Get user information
        user_data = get_user_info(provider, access_token)
        if not user_data:
            return False
        
        # Normalize user data
        normalized_user = normalize_user_data(provider, user_data)
        if not normalized_user:
            st.error("Failed to process user information")
            return False
        
        # Auto-register/login user
        username = f"{provider.lower()}_{normalized_user['username']}"
        email = normalized_user['email']
        
        # Try to register (will fail if already exists, which is fine)
        register_user(username, email, secrets.token_urlsafe(32))
        
        # Set logged in state
        st.session_state.logged_in = True
        st.session_state.username = username
        st.session_state.user_email = email
        st.session_state.user_name = normalized_user['name']
        st.session_state.oauth_provider = provider
        
        # Clear OAuth params from URL
        st.query_params.clear()
        
        return True
    
    return False

def initiate_oauth_login(provider):
    """Start the OAuth login flow for a provider"""
    config = OAUTH_PROVIDERS.get(provider)
    
    if not config:
        st.error(f"Provider {provider} is not configured")
        return
    
    if not config['enabled']:
        st.info(f"🔧 {provider} login requires API configuration.")
        st.markdown(f"""
        **Setup Instructions:**
        1. Create an OAuth app on {provider}'s developer platform
        2. Add credentials to `oauth_config.py`
        3. Set redirect URI to: `http://localhost:8501`
        
        **Quick Links:**
        - **Google**: [Console](https://console.cloud.google.com/)
        - **GitHub**: [Developer Settings](https://github.com/settings/developers)
        - **Facebook**: [Developer Portal](https://developers.facebook.com/)
        - **Microsoft**: [Azure Portal](https://portal.azure.com/)
        - **Twitter**: [Developer Portal](https://developer.twitter.com/)
        - **Apple**: [Developer Portal](https://developer.apple.com/)
        """)
        return
    
    # Build authorization URL
    auth_url = build_auth_url(provider)
    if auth_url:
        st.markdown(f'<meta http-equiv="refresh" content="0;url={auth_url}">', unsafe_allow_html=True)
        st.info(f"🔄 Redirecting to {provider} for authentication...")
    else:
        st.error(f"Failed to generate {provider} authorization URL")
