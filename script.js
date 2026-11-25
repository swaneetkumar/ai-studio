/**
 * AI Foley Studio - Interactive Scripts
 * Strict implementation of landing-page-design motion and interaction rules
 */

document.addEventListener('DOMContentLoaded', () => {
  initTaglineReveal();
  initScrollReveals();
  initStudioMockup();
  initFaqAccordion();
  initMobileMenu();
  initBackToTop();
});

/**
 * B11 Tagline reveal section
 * Words start at 30% opacity and transition to full opacity one by one as they cross the viewport trigger line
 */
function initTaglineReveal() {
  const container = document.querySelector('.tagline-reveal-section');
  const words = document.querySelectorAll('.tagline-word');
  if (!container || words.length === 0) return;

  // Reduced motion check
  const prefersReduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  if (prefersReduced) {
    words.forEach(word => word.classList.add('revealed'));
    return;
  }

  let ticking = false;

  function updateWords() {
    const rect = container.getBoundingClientRect();
    const windowHeight = window.innerHeight;

    // Calculate how far into the container the viewport has reached
    const triggerOffset = windowHeight * 0.75;
    const progress = (triggerOffset - rect.top) / (rect.height + triggerOffset * 0.5);
    const clampedProgress = Math.max(0, Math.min(1, progress));

    const totalWords = words.length;
    const activeCount = Math.floor(clampedProgress * totalWords);

    words.forEach((word, index) => {
      if (index <= activeCount && clampedProgress > 0.05) {
        word.classList.add('revealed');
      } else {
        word.classList.remove('revealed');
      }
    });

    ticking = false;
  }

  window.addEventListener('scroll', () => {
    if (!ticking) {
      window.requestAnimationFrame(updateWords);
      ticking = true;
    }
  }, { passive: true });

  updateWords();
}

/**
 * Scroll reveals using IntersectionObserver (Section B7)
 * Elements rise 32px and fade from 0 to 1 over 700ms on base curve
 */
function initScrollReveals() {
  const elements = document.querySelectorAll('.reveal-on-scroll');
  if (!elements.length) return;

  const observer = new IntersectionObserver((entries, obs) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('revealed');
        obs.unobserve(entry.target); // Reveal once only
      }
    });
  }, {
    root: null,
    rootMargin: '0px 0px -40px 0px',
    threshold: 0.15
  });

  elements.forEach(el => observer.observe(el));
}

/**
 * Interactive Studio Mockup in Hero
 * Allows switching between Raw Silent Video and AI Foley Studio
 */
function initStudioMockup() {
  const toggleRaw = document.getElementById('toggle-raw');
  const toggleFoley = document.getElementById('toggle-foley');
  const playBtn = document.getElementById('mockup-play-btn');
  const bars = document.querySelectorAll('.wave-bar');
  const sceneIndicator = document.getElementById('scene-status-text');
  const timelineLabel = document.getElementById('timeline-status-label');
  const eventRows = document.querySelectorAll('.event-item');

  if (!toggleRaw || !toggleFoley) return;

  let isFoleyMode = true;
  let isPlaying = true;
  let animationInterval = null;

  function setMode(foley) {
    isFoleyMode = foley;
    if (foley) {
      toggleFoley.classList.add('active');
      toggleRaw.classList.remove('active');
      if (sceneIndicator) sceneIndicator.textContent = 'MMAudio Foley: 44.1 kHz active';
      if (timelineLabel) timelineLabel.textContent = 'AudioLDM + MMAudio track (0.95 peak limited)';
      eventRows.forEach(row => row.classList.remove('inactive-mode'));
      startWaveform();
    } else {
      toggleRaw.classList.add('active');
      toggleFoley.classList.remove('active');
      if (sceneIndicator) sceneIndicator.textContent = 'Silent input: no audio track';
      if (timelineLabel) timelineLabel.textContent = 'Muted original video channel';
      eventRows.forEach(row => row.classList.add('inactive-mode'));
      stopWaveform();
    }
  }

  toggleRaw.addEventListener('click', () => setMode(false));
  toggleFoley.addEventListener('click', () => setMode(true));

  function startWaveform() {
    if (animationInterval) clearInterval(animationInterval);
    animationInterval = setInterval(() => {
      bars.forEach(bar => {
        const heightPercent = isFoleyMode ? Math.floor(Math.random() * 80 + 20) : 4;
        bar.style.height = `${heightPercent}%`;
      });
    }, 120);
  }

  function stopWaveform() {
    if (animationInterval) clearInterval(animationInterval);
    bars.forEach(bar => {
      bar.style.height = '4%';
    });
  }

  if (playBtn) {
    playBtn.addEventListener('click', () => {
      isPlaying = !isPlaying;
      if (isPlaying && isFoleyMode) {
        startWaveform();
      } else {
        stopWaveform();
      }
    });
  }

  startWaveform();
}


/**
 * Accessible FAQ Accordion
 */
function initFaqAccordion() {
  const faqItems = document.querySelectorAll('.faq-item');
  faqItems.forEach(item => {
    const questionBtn = item.querySelector('.faq-question');
    if (!questionBtn) return;

    questionBtn.addEventListener('click', () => {
      const isOpen = item.classList.contains('open');

      // Close other accordion items for clean single expansion
      faqItems.forEach(otherItem => {
        if (otherItem !== item) {
          otherItem.classList.remove('open');
          const btn = otherItem.querySelector('.faq-question');
          if (btn) btn.setAttribute('aria-expanded', 'false');
        }
      });

      if (isOpen) {
        item.classList.remove('open');
        questionBtn.setAttribute('aria-expanded', 'false');
      } else {
        item.classList.add('open');
        questionBtn.setAttribute('aria-expanded', 'true');
      }
    });
  });
}

/**
 * Mobile Navigation Toggle with Staggered Entrance
 */
function initMobileMenu() {
  const menuToggle = document.querySelector('.menu-toggle');
  const mobileMenu = document.querySelector('.mobile-menu');
  const mobileLinks = document.querySelectorAll('.mobile-nav-link');

  if (!menuToggle || !mobileMenu) return;

  function toggleMenu() {
    const isOpen = mobileMenu.classList.contains('open');
    if (isOpen) {
      mobileMenu.classList.remove('open');
      menuToggle.setAttribute('aria-expanded', 'false');
      document.body.style.overflow = '';
    } else {
      mobileMenu.classList.add('open');
      menuToggle.setAttribute('aria-expanded', 'true');
      document.body.style.overflow = 'hidden';

      // Stagger items by 60ms, cap at 400ms
      const items = mobileMenu.querySelectorAll('.mobile-nav-links li');
      items.forEach((item, index) => {
        const delay = Math.min(index * 60, 400);
        item.style.transitionDelay = `${delay}ms`;
      });
    }
  }

  menuToggle.addEventListener('click', toggleMenu);

  mobileLinks.forEach(link => {
    link.addEventListener('click', () => {
      mobileMenu.classList.remove('open');
      menuToggle.setAttribute('aria-expanded', 'false');
      document.body.style.overflow = '';
    });
  });
}

/**
 * Back to top link
 */
function initBackToTop() {
  const btn = document.querySelector('.back-to-top');
  if (!btn) return;
  btn.addEventListener('click', () => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  });
}

