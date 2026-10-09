import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

const config = window.SUPABASE_CONFIG || {};
const configured = typeof config.url === "string" &&
  config.url.startsWith("https://") &&
  !config.url.includes("PASTE_YOUR") &&
  typeof config.publishableKey === "string" &&
  config.publishableKey.length > 10 &&
  !config.publishableKey.includes("PASTE_YOUR");

const gate = document.getElementById("authGate");
const message = document.getElementById("authMessage");
const loginForm = document.getElementById("loginForm");
const setPasswordForm = document.getElementById("setPasswordForm");
const forgotButton = document.getElementById("forgotPasswordButton");
const logoutButton = document.getElementById("logoutButton");
let supabase = null;

function showMessage(text, kind = "") {
  message.textContent = text;
  message.className = "auth-message" + (kind ? " " + kind : "");
}
function showGate() {
  document.body.classList.add("auth-pending");
  gate.style.display = "";
}
function showWorkspace(user) {
  document.body.classList.remove("auth-pending");
  gate.style.display = "none";
  if (user?.email) {
    const profile = document.querySelector(".profile-copy strong");
    const topProfile = document.getElementById("topProfile");
    const avatar = document.querySelector(".profile-button .avatar");
    if (profile) profile.textContent = user.email;
    const initials = user.email.split("@")[0].slice(0, 2).toUpperCase();
    if (topProfile) topProfile.textContent = initials;
    if (avatar) avatar.textContent = initials;
  }
}
function isPasswordSetupLink() {
  const params = new URLSearchParams(window.location.search);
  const hash = new URLSearchParams(window.location.hash.replace(/^#/, ""));
  return ["invite", "recovery", "signup"].includes(params.get("type")) ||
    ["invite", "recovery", "signup"].includes(hash.get("type")) ||
    params.has("code") ||
    window.location.hash.includes("type=recovery") ||
    window.location.hash.includes("type=invite");
}

if (!configured) {
  showGate();
  showMessage("Setup required: add your Supabase Project URL and Publishable key in saas-frontend/supabase-config.js, then deploy this preview branch.", "error");
  loginForm.querySelector("button[type=submit]").disabled = true;
  forgotButton.disabled = true;
} else {
  supabase = createClient(config.url, config.publishableKey, {
    auth: { persistSession: true, autoRefreshToken: true, detectSessionInUrl: true }
  });

  async function syncSession() {
    const { data, error } = await supabase.auth.getSession();
    if (error) {
      showGate();
      showMessage("Could not check your session: " + error.message, "error");
      return;
    }
    if (data.session?.user) {
      if (isPasswordSetupLink()) {
        showGate();
        loginForm.classList.add("hidden");
        forgotButton.classList.add("hidden");
        setPasswordForm.classList.remove("hidden");
        showMessage("Invitation or password reset link verified. Please set your password.");
      } else {
        showWorkspace(data.session.user);
      }
    } else {
      showGate();
      showMessage("Sign in with the email address used for your invitation.");
    }
  }

  loginForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const email = document.getElementById("loginEmail").value.trim();
    const password = document.getElementById("loginPassword").value;
    const button = document.getElementById("loginButton");
    button.disabled = true;
    showMessage("Signing in…");
    const { data, error } = await supabase.auth.signInWithPassword({ email, password });
    button.disabled = false;
    if (error) {
      showMessage("Sign-in failed. Check your email/password and confirm this account was invited by the admin. " + error.message, "error");
      return;
    }
    if (data.session?.user) {
      showWorkspace(data.session.user);
      showMessage("Signed in successfully.", "success");
    }
  });

  forgotButton.addEventListener("click", async () => {
    const email = document.getElementById("loginEmail").value.trim();
    if (!email) {
      showMessage("Enter your email address first, then choose Forgot password.", "error");
      document.getElementById("loginEmail").focus();
      return;
    }
    showMessage("Sending password reset email…");
    const { error } = await supabase.auth.resetPasswordForEmail(email, {
      redirectTo: window.location.origin + window.location.pathname
    });
    if (error) showMessage("Could not send reset email: " + error.message, "error");
    else showMessage("If this email belongs to an account, a password reset link will be sent.", "success");
  });

  setPasswordForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const password = document.getElementById("newPassword").value;
    const confirm = document.getElementById("confirmPassword").value;
    if (password.length < 8) {
      showMessage("Use at least 8 characters for the password.", "error");
      return;
    }
    if (password !== confirm) {
      showMessage("The passwords do not match.", "error");
      return;
    }
    const { error } = await supabase.auth.updateUser({ password });
    if (error) {
      showMessage("Could not save password: " + error.message, "error");
      return;
    }
    window.history.replaceState({}, document.title, window.location.pathname);
    loginForm.classList.remove("hidden");
    forgotButton.classList.remove("hidden");
    setPasswordForm.classList.add("hidden");
    await supabase.auth.signOut();
    showGate();
    showMessage("Password saved. You can now sign in.", "success");
  });

  logoutButton?.addEventListener("click", async () => {
    const { error } = await supabase.auth.signOut();
    if (error) {
      showMessage("Sign out failed: " + error.message, "error");
      return;
    }
    loginForm.reset();
    showGate();
    showMessage("You have signed out.");
  });

  supabase.auth.onAuthStateChange((event, session) => {
    if (session?.user && (event === "PASSWORD_RECOVERY" || isPasswordSetupLink())) {
      showGate();
      loginForm.classList.add("hidden");
      forgotButton.classList.add("hidden");
      setPasswordForm.classList.remove("hidden");
      showMessage("Invitation or password reset link verified. Please set your password.");
    } else if (session?.user) {
      showWorkspace(session.user);
    } else if (!session) {
      showGate();
    }
  });

  syncSession();
}
