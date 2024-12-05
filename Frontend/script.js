document.addEventListener('DOMContentLoaded', function() {
  const accountLink = document.getElementById('account-link');
  const modal = document.getElementById('auth-modal');
  const closeBtn = document.querySelector('.close-btn');
  const switchToRegister = document.getElementById('switch-to-register');
  const switchToLogin = document.getElementById('switch-to-login');
  const loginForm = document.getElementById('login-form');
  const registerForm = document.getElementById('register-form');

  // Show modal
  accountLink.addEventListener('click', function() {
      modal.style.display = 'block';
  });

  // Close modal
  closeBtn.addEventListener('click', function() {
      modal.style.display = 'none';
  });

  window.addEventListener('click', function(event) {
      if (event.target === modal) {
          modal.style.display = 'none';
      }
  });

  // Switch to register form
  switchToRegister.addEventListener('click', function(e) {
      e.preventDefault();
      loginForm.style.display = 'none';
      registerForm.style.display = 'block';
  });

  // Switch to login form
  switchToLogin.addEventListener('click', function(e) {
      e.preventDefault();
      registerForm.style.display = 'none';
      loginForm.style.display = 'block';
  });
});
