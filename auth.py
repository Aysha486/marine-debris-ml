import json
import os
import hashlib
from pathlib import Path
from datetime import datetime

# Database file for users
USERS_DB = Path(__file__).resolve().parent / 'users.json'

def hash_password(password: str) -> str:
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def load_users() -> dict:
    """Load users from JSON database"""
    if USERS_DB.exists():
        try:
            with open(USERS_DB, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_users(users: dict):
    """Save users to JSON database"""
    with open(USERS_DB, 'w') as f:
        json.dump(users, f, indent=2)

def user_exists(username: str) -> bool:
    """Check if username already exists"""
    users = load_users()
    return username.lower() in users

def register_user(username: str, email: str, password: str) -> tuple[bool, str]:
    """
    Register a new user
    Returns: (success, message)
    """
    # Validation
    if not username or len(username) < 3:
        return False, "Username must be at least 3 characters"
    
    if not email or '@' not in email:
        return False, "Invalid email address"
    
    if not password or len(password) < 6:
        return False, "Password must be at least 6 characters"
    
    # Check if user exists
    if user_exists(username):
        return False, "Username already exists"
    
    # Create user
    users = load_users()
    users[username.lower()] = {
        'email': email,
        'password': hash_password(password),
        'created_at': datetime.now().isoformat(),
        'last_login': None
    }
    save_users(users)
    
    return True, "Registration successful! You can now login."

def login_user(username: str, password: str) -> tuple[bool, str]:
    """
    Authenticate user
    Returns: (success, message)
    """
    users = load_users()
    username_lower = username.lower()
    
    if username_lower not in users:
        return False, "Username not found"
    
    user = users[username_lower]
    if user['password'] != hash_password(password):
        return False, "Incorrect password"
    
    # Update last login
    user['last_login'] = datetime.now().isoformat()
    save_users(users)
    
    return True, "Login successful!"

def get_user_info(username: str) -> dict:
    """Get user information"""
    users = load_users()
    return users.get(username.lower(), {})
