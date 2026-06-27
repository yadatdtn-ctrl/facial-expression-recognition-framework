document.addEventListener('DOMContentLoaded', () => {
  const githubUrl = 'https://github.com/yadatdtn-ctrl/facial-expression-recognition-framework';
  const kaggleUrl = 'https://www.kaggle.com/code/yadaaaat/facial-expression-recognition-framework';
  const demoUrl = 'https://yada-ctrl-facexplain-demo.hf.space';

  const navItems = [
    { label: 'Home', href: 'index.html' },
    { label: 'Demo', href: 'demo.html' },
    { label: 'GitHub', href: githubUrl, external: true }
  ];

  const navLinks = document.querySelector('.nav-links');
  if (navLinks) {
    navLinks.innerHTML = navItems
      .map((item) => {
        if (item.external) {
          return `<li><a href="${item.href}" target="_blank">${item.label}</a></li>`;
        }
        return `<li><a href="${item.href}">${item.label}</a></li>`;
      })
      .join('');
  }

  document.querySelectorAll('a').forEach((link) => {
    const text = link.textContent.trim();
    if ((text === 'GitHub' || text === 'View on GitHub') && link.getAttribute('href') === '#') {
      link.href = githubUrl;
      link.target = '_blank';
    }
    if (text === 'Kaggle' && link.getAttribute('href') === '#') {
      link.href = kaggleUrl;
      link.target = '_blank';
    }
  });

  const demoBtn = document.querySelector('.nav-cta');
  if (demoBtn && demoBtn.textContent.trim() === 'Try the demo') {
    demoBtn.removeAttribute('onclick');
    demoBtn.addEventListener('click', () => {
      window.open(demoUrl, '_blank', 'noopener,noreferrer');
    });
  }

  const pathname = window.location.pathname;
  const currentPage = pathname.split('/').pop() || 'index.html';

  document.querySelectorAll('.nav-links a').forEach((link) => {
    const href = link.getAttribute('href');
    if (!href || href.startsWith('#') || href.startsWith('http')) return;

    const linkPage = href.split('/').pop();
    if (linkPage === currentPage || (currentPage === '' && linkPage === 'index.html')) {
      link.classList.add('active');
    }
  });

  document.querySelectorAll('.footer-col-links a').forEach((link) => {
    const href = link.getAttribute('href');
    if (!href || href.startsWith('http')) return;

    const linkPage = href.split('/').pop();
    if (linkPage === currentPage || (currentPage === '' && linkPage === 'index.html')) {
      link.classList.add('active');
    }
  });

  document.querySelectorAll('.site-footer a[href="#"]').forEach((link) => {
    const text = link.textContent.trim();
    if (text === 'GitHub') {
      link.href = githubUrl;
      link.target = '_blank';
      link.rel = 'noopener noreferrer';
    }
    if (text === 'Kaggle') {
      link.href = kaggleUrl;
      link.target = '_blank';
      link.rel = 'noopener noreferrer';
    }
  });

  const hamburger = document.querySelector('.hamburger');

  if (hamburger && navLinks) {
    hamburger.addEventListener('click', () => {
      navLinks.classList.toggle('open');
    });
  }

  document.querySelectorAll('a[href^="#"]').forEach((anchor) => {
    anchor.addEventListener('click', (event) => {
      const targetId = anchor.getAttribute('href');
      if (!targetId || targetId === '#') return;

      const target = document.querySelector(targetId);
      if (!target) return;

      event.preventDefault();
      target.scrollIntoView({ behavior: 'smooth' });
    });
  });
});
