/* FakeGuard — Enhanced Frontend */
document.addEventListener('DOMContentLoaded', () => {
    createParticles();
    initGauge();
    initCharCounter();
    initPasswordStrength();
    initNavToggle();
    initFormLoaders();
    animateCards();
    initTypingEffect();
    initMouseGlow();
});

function createParticles() {
    const c = document.getElementById('bg-particles');
    if (!c) return;
    const colors = ['#00d4ff','#9b4dff','#ff4da6','#23d160','#00d4ff'];
    for (let i = 0; i < 35; i++) {
        const p = document.createElement('div');
        p.className = 'particle';
        const size = Math.random() * 5 + 1.5;
        const color = colors[Math.floor(Math.random()*colors.length)];
        p.style.cssText = `width:${size}px;height:${size}px;left:${Math.random()*100}%;background:${color};box-shadow:0 0 ${size*3}px ${color};animation-duration:${Math.random()*20+12}s;animation-delay:${Math.random()*12}s;opacity:0;`;
        c.appendChild(p);
    }
}

function initGauge() {
    const circle = document.getElementById('gauge-circle');
    const valueEl = document.getElementById('gauge-value');
    if (!circle || !valueEl) return;
    const confidence = parseFloat(circle.dataset.confidence);
    const circumference = 2 * Math.PI * 85;
    const offset = circumference - (confidence / 100) * circumference;
    setTimeout(() => {
        circle.style.strokeDashoffset = offset;
        animateCounter(valueEl, 0, confidence, 1800);
    }, 400);
}

function animateCounter(el, start, end, duration) {
    const startTime = performance.now();
    function update(t) {
        const p = Math.min((t - startTime) / duration, 1);
        const eased = 1 - Math.pow(1 - p, 4);
        el.textContent = Math.round(start + (end - start) * eased);
        if (p < 1) requestAnimationFrame(update);
    }
    requestAnimationFrame(update);
}

function initCharCounter() {
    const ta = document.getElementById('headline');
    const ct = document.getElementById('char-count');
    if (!ta || !ct) return;
    ta.addEventListener('input', () => { ct.textContent = ta.value.length; });
}

function initPasswordStrength() {
    const pw = document.getElementById('password');
    const fill = document.getElementById('strength-fill');
    const text = document.getElementById('strength-text');
    if (!pw || !fill || !text) return;
    pw.addEventListener('input', () => {
        const v = pw.value; let s = 0;
        if (v.length >= 6) s++;
        if (v.length >= 10) s++;
        if (/[A-Z]/.test(v)) s++;
        if (/[0-9]/.test(v)) s++;
        if (/[^A-Za-z0-9]/.test(v)) s++;
        const colors = ['#ff3860','#ff6b35','#ffaa00','#a0e634','#23d160'];
        const labels = ['Very Weak','Weak','Fair','Strong','Very Strong'];
        fill.style.width = (s / 5) * 100 + '%';
        fill.style.background = colors[Math.max(s-1,0)];
        text.textContent = s > 0 ? labels[s-1] : 'At least 6 characters';
    });
}

function togglePassword(id) {
    const el = document.getElementById(id);
    if (el) el.type = el.type === 'password' ? 'text' : 'password';
}

function initNavToggle() {
    const btn = document.getElementById('nav-toggle');
    if (!btn) return;
    btn.addEventListener('click', () => {
        document.querySelector('.nav-links')?.classList.toggle('open');
    });
}

function initFormLoaders() {
    document.querySelectorAll('form').forEach(f => {
        f.addEventListener('submit', () => {
            const btn = f.querySelector('.btn-primary');
            if (!btn) return;
            const t = btn.querySelector('.btn-text');
            const l = btn.querySelector('.btn-loader');
            if (t) t.textContent = 'Analyzing...';
            if (l) l.style.display = 'inline-block';
            btn.disabled = true;
            btn.style.opacity = '0.7';
        });
    });
}

function animateCards() {
    const obs = new IntersectionObserver((entries) => {
        entries.forEach((e, i) => {
            if (e.isIntersecting) {
                setTimeout(() => {
                    e.target.style.opacity = '1';
                    e.target.style.transform = 'translateY(0)';
                }, i * 80);
            }
        });
    }, { threshold: 0.05 });
    document.querySelectorAll('.glass-card,.info-card,.history-item').forEach(el => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(25px)';
        el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        obs.observe(el);
    });
}

function initTypingEffect() {
    const ta = document.getElementById('headline');
    if (!ta || ta.value) return;
    const hint = 'Try: "BREAKING: Scientists confirm Earth is flat!!!"';
    let i = 0;
    const iv = setInterval(() => {
        if (i >= hint.length || document.activeElement === ta) { clearInterval(iv); ta.placeholder = 'Paste a news headline or article text here...\n\nExample: Scientists discover new species in the Amazon rainforest'; return; }
        ta.setAttribute('placeholder', hint.substring(0, ++i) + '|');
    }, 50);
}

function initMouseGlow() {
    const cards = document.querySelectorAll('.analyzer-card, .result-card');
    cards.forEach(card => {
        card.addEventListener('mousemove', (e) => {
            const r = card.getBoundingClientRect();
            const x = e.clientX - r.left;
            const y = e.clientY - r.top;
            card.style.setProperty('--mx', x + 'px');
            card.style.setProperty('--my', y + 'px');
            card.style.background = `radial-gradient(600px circle at ${x}px ${y}px, rgba(0,212,255,0.03), transparent 40%), rgba(14,14,28,0.75)`;
        });
        card.addEventListener('mouseleave', () => {
            card.style.background = 'rgba(14,14,28,0.75)';
        });
    });
}
