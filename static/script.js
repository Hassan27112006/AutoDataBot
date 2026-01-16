// static/script.js
const chatBox = document.getElementById('chat-box');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const analyzeBtn = document.getElementById('analyze-btn');

function appendMessage(sender, html, meta = '') {
  const msg = document.createElement('div');
  msg.className = 'message ' + sender;
  msg.innerHTML = `
    <div class="bubble">${html}</div>
    <div class="meta">${meta}</div>
  `;
  chatBox.appendChild(msg);
  chatBox.scrollTop = chatBox.scrollHeight;
  return msg;               // ⭐ Return message DOM element so we can update meta later
}

function showTyping() {
  const t = document.createElement('div');
  t.id = 'typing';
  t.className = 'message bot';
  t.innerHTML = `
    <div class="bubble typing">
      <span class="dot"></span><span class="dot"></span><span class="dot"></span>
    </div>`;
  chatBox.appendChild(t);
  chatBox.scrollTop = chatBox.scrollHeight;
}

function hideTyping() {
  const t = document.getElementById('typing');
  if (t) t.remove();
}

sendBtn.onclick = async () => {
  const text = userInput.value.trim();
  if (!text) return;

  appendMessage('user', escapeHtml(text));
  userInput.value = '';

  showTyping();

  const res = await fetch('/chat', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ message: text })
  });

  const data = await res.json();
  hideTyping();

  appendMessage('bot', escapeHtml(data.response));
};

analyzeBtn.onclick = () => {
  const input = document.createElement('input');
  input.type = 'file';
  input.accept = '.csv, .xlsx';

  input.onchange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    // Show file message with uploading...
    const msg = appendMessage(
      'user',
      `📂 ${escapeHtml(file.name)}`,
      'uploading...'
    );

    showTyping();

    const fd = new FormData();
    fd.append('file', file);

    const res = await fetch('/analyze', {
      method: 'POST',
      body: fd
    });

    const data = await res.json();
    hideTyping();

    // ⭐ Change uploading... → animated green checkmark ✔️
    const meta = msg.querySelector('.meta');
    meta.classList.add('success');
    meta.innerHTML = `<span class="checkmark"></span> Uploaded`;

    if (data.error) {
      appendMessage('bot', `Error: ${escapeHtml(data.error)}`);
    } else {
      appendMessage(
        'bot',
        `Analysis finished.<br>Download ZIP: <a href="/download/${escapeHtml(data.zip)}">${escapeHtml(data.zip)}</a>`
      );
    }
  };

  input.click();
};

function escapeHtml(unsafe) {
  if (!unsafe) return '';
  return unsafe.replace(/[&<"'>]/g, function (m) {
    return {
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      '"': '&quot;',
      "'": '&#039;'
    }[m];
  });
}
