import base64, json, streamlit as st
from streamlit_oauth import OAuth2Component

def google_oauth_login():
    # Check if user is already authenticated
    if "user_email" in st.session_state and "authenticated" in st.session_state:
        if st.session_state["authenticated"]:
            st.success(f"Already logged in as {st.session_state['user_email']}")
            return st.session_state["user_email"]
    
    oauth = OAuth2Component(
        client_id     = st.secrets["oauth_client_id"],
        client_secret = st.secrets["oauth_client_secret"],
        authorize_endpoint = "https://accounts.google.com/o/oauth2/v2/auth",
        token_endpoint     = "https://oauth2.googleapis.com/token",
    )

    try:
        result = oauth.authorize_button(
            name="Login with Google",
            icon="🔐",
            redirect_uri = st.secrets["redirect_uri"],
            scope="openid email profile",
            key="google",           # optional but avoids duplicate buttons
            pkce="S256",            # recommended for public clients
        )
    except Exception as e:
        st.error(f"Authentication error: {str(e)}")
        st.info("Please refresh the page and try again.")
        if st.button("🔄 Clear Session & Retry"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.experimental_rerun()
        return None

    if not result:                         # user has not finished the flow yet
        return None

    try:
        id_token = result["token"]["id_token"] # JWT issued by Google

        # -- decode the JWT payload (middle part) without verification -------------
        payload_b64 = id_token.split(".")[1] + "==="          # pad for base64
        payload = json.loads(base64.urlsafe_b64decode(payload_b64))
        email   = payload["email"]

        st.session_state["user_email"] = email
        st.session_state["authenticated"] = True
        st.success(f"Logged in as {email}")
        return email
    except Exception as e:
        st.error(f"Error processing authentication token: {str(e)}")
        return None
