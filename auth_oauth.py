import streamlit as st

# Title of your app
st.set_page_config(page_title="CyberShield ML", layout="centered")
st.title("🔒 CyberShield ML – Secure Access")

# Google OAuth2 Login function using native Streamlit support
def google_oauth_login():
    if "user_email" in st.session_state:
        st.success(f"✅ Already logged in as: {st.session_state['user_email']}")
        return st.session_state["user_email"]

    login_info = st.oauth2_login(
        client_id=st.secrets["oauth_client_id"],
        client_secret=st.secrets["oauth_client_secret"],
        token_endpoint="https://oauth2.googleapis.com/token",
        authorize_endpoint="https://accounts.google.com/o/oauth2/v2/auth",
        redirect_uri=st.secrets["redirect_uri"],
        scope="openid email profile",
        provider_name="Google"
    )

    if login_info:
        st.session_state["user_email"] = login_info["user_info"]["email"]
        st.success(f"✅ Logged in as: {st.session_state['user_email']}")
        return st.session_state["user_email"]
    else:
        st.info("🔐 Please log in with your Google account.")
        return None

# Main app logic
def main():
    user = google_oauth_login()
    
    if user:
        st.subheader("🚀 Welcome to CyberShield ML")
        st.write("You are now authenticated to access the application.")
        # You can now show rest of the app: model predictions, file upload, visualizations, etc.
    else:
        st.stop()  # Prevent the rest of the app from showing

if __name__ == "__main__":
    main()
